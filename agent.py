"""VIT-AP Outbound Verification Voice Agent.

This module sets up an outbound automated voice calling agent using LiveKit
Agents, Deepgram STT, OpenRouter (via the OpenAI LLM adapter), and OpenAI TTS.
It establishes a SIP call session, parses student metadata, and runs an
interactive verification pipeline with voice activity detection (VAD).
"""

import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    VoicePipelineAgent,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, openai, silero

load_dotenv()
logger = logging.getLogger("VITAP-Outbound-Agent")
logger.setLevel(logging.INFO)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not defined in your environment.")


async def entrypoint(ctx: JobContext) -> None:
    """Execute the core lifecycle for the outbound agent job.

    Connects to the designated LiveKit room, waits for the recipient SIP
    participant, parses application metadata, configures the speech-to-speech
    pipeline, and delivers the dynamic greeting.

    Args:
        ctx (JobContext): Context object holding room session state and
            dispatch lifecycle handlers.

    Returns:
        None
    """
    logger.info("Connecting agent to room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 1. Wait for SIP participant (the student) to join the call
    participant = await ctx.wait_for_participant()
    logger.info(
        "Participant joined: identity=%s, name=%s",
        participant.identity,
        participant.name,
    )

    # 2. Extract context passed from make_call.py
    metadata = {}
    if participant.metadata:
        try:
            metadata = json.loads(participant.metadata)
        except Exception:
            metadata = {}

    student_name = (
        metadata.get("student_name") or participant.name or "Student"
    )
    app_number = metadata.get("application_number", "N/A")
    issue_transcription = metadata.get(
        "issue_transcription", "general inquiry"
    )

    # 3. Formulate the dynamic verification prompt
    system_prompt = (
        "You are a polite, helpful, and concise verification assistant "
        "calling on behalf of VIT-AP University.\n"
        f"You are speaking with {student_name} regarding their Application "
        f"Number: {app_number}.\n"
        f'Their documented support record states: "{issue_transcription}".\n\n'
        "Your primary goal:\n"
        f"1. Verify if {student_name} wishes to continue with their "
        "application"
        " / issue resolution.\n"
        "2. Confirm their response clearly.\n"
        "3. Be courteous, clear, and concise.\n"
        "4. Once they confirm whether they want to proceed or cancel, "
        "acknowledge their answer and politely wrap up the call."
    )

    # 4. OpenRouter configuration via OpenAI plugin adapter
    openrouter_llm = openai.LLM.with_openrouter(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.3,
    )

    initial_ctx = llm.ChatContext()
    initial_ctx.append(role="system", text=system_prompt)

    # 5. Build voice assistant pipeline with VoicePipelineAgent
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2", language="en-IN"),
        llm=openrouter_llm,
        tts=openai.TTS(voice="alloy"),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)

    # 6. Spoken greeting when recipient answers
    greeting = (
        f"Hello {student_name}! I am calling from VIT-AP University regarding "
        f"your application number {app_number}. We have noted your issue "
        f"regarding '{issue_transcription}'. Are you still looking to proceed "
        "with this?"
    )
    await agent.say(greeting, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
