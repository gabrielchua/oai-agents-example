"""
run.py

This is a multi-agent workflow that uses a query router to determine which agent to use based on the user's query.
It ends with a reply agent that formats the response with emojis.

There are multiple input and output guardrails to check the input and output of the agents.
"""

# Standard Library Imports
import asyncio
import sys
import time
from typing import Literal

# Third Party Imports
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    WebSearchTool,
    input_guardrail,
    output_guardrail,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

WORKFLOW_NAME = "Travel Query Workflow"
GROUP_ID = "conversation-1"
USER_ID = "123"


# ------------------------------------------------------------------------------------------------
# Initialize OpenAI client
# ------------------------------------------------------------------------------------------------

client = AsyncOpenAI()

# ------------------------------------------------------------------------------------------------
# Response Schema
# ------------------------------------------------------------------------------------------------

class MessageOutput(BaseModel): 
    response: str

class MessageOutputWithCoT(BaseModel):
    reasoning: str
    response: str

class UrgencyInput(BaseModel):
    urgency: Literal["urgent", "not urgent"]

class RelevanceInputGuardrailOutput(BaseModel):
    reasoning: str = Field(description="Use this as a scratchpad to reflect for whether the input is relevant to travelling.")
    is_irrelevant: bool = Field(description="Your final answer.")
    error_message: Literal["Input is relevant", "Input is irrelevant"] = Field(description="If the input is relevant to travelling, return 'Input is relevant'. If the input is not relevant to travelling, return 'Input is irrelevant'.")

class MinLengthInputGuardrailOutput(BaseModel):
    is_too_short: bool = Field(description="Your final answer.")
    error_message: Literal["Input is long enough", "Input is too short"] = Field(description="If the input is long enough, return 'Input is long enough'. If the input is too short, return 'Input is too short'.")

class ModerationInputGuardrailOutput(BaseModel):
    is_flagged: bool = Field(description="Your final answer.")
    error_message: Literal["Input is not flagged", "Input is flagged"] = Field(description="If the input is not flagged, return 'Input is not flagged'. If the input is flagged, return 'Input is flagged'.")

class OutputGuardrailOutput(BaseModel):
    reasoning: str = Field(description="Use this as a scratchpad to reflect for whether the output contains any non-English content.")
    is_non_english: bool = Field(description="Your final answer.")
    error_message: Literal["Output is non-English", "Output is English"] = Field(description="If the output contains any non-English content, return 'Output is non-English'. If the output contains only English content, return 'Output is English'.")


# ------------------------------------------------------------------------------------------------
# Guardrail agents
# ------------------------------------------------------------------------------------------------

input_guardrail_agent = Agent( 
    name="Guardrail check",
    model="gpt-4o-mini-2024-07-18",
    instructions="Check if the user is asking you something that is not related to travelling.",
    output_type=RelevanceInputGuardrailOutput,
)

output_guardrail_agent = Agent( 
    name="Guardrail check",
    model="gpt-4o-mini-2024-07-18",
    instructions="Check if the output contains any non-English content.",
    output_type=OutputGuardrailOutput,
)


@input_guardrail
async def relevance_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_irrelevant,
    )

@input_guardrail
async def min_length_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    user_messages = [message['content'] for message in input if message['role'] == 'user']
    latest_user_message = user_messages[-1]
    input_length = len(latest_user_message)
    if input_length < 10:
        return GuardrailFunctionOutput(
            output_info=MinLengthInputGuardrailOutput(is_too_short=True, error_message="Input is too short"),
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(output_info=MinLengthInputGuardrailOutput(is_too_short=False, error_message="Input is long enough"), tripwire_triggered=False)


@input_guardrail
async def moderation_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    user_messages = [message['content'] for message in input if message['role'] == 'user']
    latest_user_message = user_messages[-1]

    response = await client.moderations.create(
        model="omni-moderation-2024-09-26",
        input=latest_user_message,
    )

    flagged = response.results[0].flagged

    if flagged:
        return GuardrailFunctionOutput(
            output_info=ModerationInputGuardrailOutput(is_flagged=flagged, error_message="Input is flagged"),
            tripwire_triggered=flagged,
        )
    return GuardrailFunctionOutput(output_info=ModerationInputGuardrailOutput(is_flagged=flagged, error_message="Input is not flagged"), tripwire_triggered=flagged)

@output_guardrail
async def non_english_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_non_english,
    )

# ------------------------------------------------------------------------------------------------
# Specialized Agents
# ------------------------------------------------------------------------------------------------
    
booking_agent = Agent(  
    name="Booking Specialist",
    model="gpt-4o-mini-2024-07-18",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a booking specialist. You help customers with their booking and reservation questions.",
    output_type=MessageOutputWithCoT,
)

travel_recommendation_agent = Agent(
    name="Travel Recommendation Specialist",
    model="gpt-4o-mini-2024-07-18",
    model_settings=ModelSettings(
        tool_choice='auto',
    ),
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a travel recommendation specialist. You help customers find ideal destinations and travel plans.",
    tools=[WebSearchTool()],
    output_type=MessageOutputWithCoT,
)

# ------------------------------------------------------------------------------------------------
# Orchestrator Agent (Tool-based approach)
# ------------------------------------------------------------------------------------------------

reply_agent = Agent(
    name="Reply Agent",
    model="gpt-4o-mini-2024-07-18",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You reply to the user's query and make it more informal by adding emojis.",
    output_type=MessageOutput,
    output_guardrails=[non_english_guardrail],
)

query_router_agent = Agent(
    name="Query Router",
    model="gpt-4o-mini-2024-07-18",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} You determine which agent to use based on the user's query. "
        "If the query relates to booking flights, use the booking specialist. "
        "If the query relates to travel recommendations, use the travel recommendation specialist. "
        "Once you get the specialist response, always hand it off to the reply agent to format it with emojis."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="consult_booking_specialist",
            tool_description="Use when the user has questions about flight bookings, reservations, or ticketing",
        ),
        travel_recommendation_agent.as_tool(
            tool_name="consult_travel_recommendation_specialist",
            tool_description="Use when the user wants travel destination recommendations or itinerary planning",
        )
    ],
    output_type=MessageOutput,
    handoffs=[reply_agent],
    input_guardrails=[relevance_guardrail, min_length_guardrail, moderation_guardrail],
)

async def main():
    # Get question from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python app.py 'Your question here'")
        return
    
    question = sys.argv[1]
    
    try:
        start_time = time.time()
        
        print("🔍 Processing your query: ", question)
        print("=" * 80)
        
        result = Runner.run_streamed(
            starting_agent=query_router_agent, 
            input=question,
            run_config=RunConfig(
                workflow_name=WORKFLOW_NAME,
                group_id=GROUP_ID,
                trace_metadata={
                    "user_id": USER_ID
                    },
                ),
            )
        
        async for event in result.stream_events():
            pass
            if event.type == "raw_response_event":
                event_data = event.data
                if isinstance(event_data, ResponseCreatedEvent):
                    agent_name = result.last_agent.name
                    print(f"🏃 Starting `{agent_name}`")
                    print("-" * 50)
                elif isinstance(event_data, ResponseInProgressEvent):
                    print("⏳ Agent response in progress...")
                elif isinstance(event_data, ResponseOutputItemAddedEvent):
                    event_data_item = event_data.item
                    if isinstance(event_data_item, ResponseFunctionToolCall):
                        print(f"🔧 Tool called: {event_data_item.name}")
                        print("\t Arguments: ", end="")
                    elif isinstance(event_data_item, ResponseOutputMessage):
                        print("📝 Drafting response...")
                elif isinstance(event_data, ResponseFunctionCallArgumentsDeltaEvent):
                    event_data_delta = event_data.delta
                    print(event_data_delta, end="", flush=True)
                elif isinstance(event_data, ResponseFunctionCallArgumentsDoneEvent):
                    print("\n✅ Tool call completed!")
                elif isinstance(event_data, ResponseTextDeltaEvent):
                    print(event_data.delta, end="", flush=True)
            elif event.type == "run_item_stream_event":
                if event.name == "tool_output":
                    print("🛠️ Tool output:")
                    print("-" * 40)
                    print(event.item.output)
                    print("-" * 40)

    except InputGuardrailTripwireTriggered as e:
        print("")
        print("=" * 60)
        print("\n🚨 INPUT GUARDRAIL TRIGGERED")
        print("=" * 60)
        guardrail_output = e.guardrail_result.output.output_info
        print(f"Error: {guardrail_output.error_message}")

    except OutputGuardrailTripwireTriggered as e:
        print("")
        print("=" * 60)
        print("\n🚨 OUTPUT GUARDRAIL TRIGGERED")
        print("=" * 60)
        guardrail_output = e.guardrail_result.output.output_info
        print(f"Error: {guardrail_output.error_message}")

    end_time = time.time()
    print("\n" + "=" * 80)
    print(f"✨ Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
