# from dotenv import load_dotenv

# from livekit import agents
# from livekit.agents import AgentSession, Agent, RoomInputOptions
# from livekit.plugins import (
#     openai,
#     # noise_cancellation,
# )
# from livekit.plugins.openai.realtime import RealtimeModel

# load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(instructions="You are a helpful voice AI assistant.")


# async def entrypoint(ctx: agents.JobContext):
#     session = AgentSession(
#         llm=openai.realtime.RealtimeModel(
#             voice="coral",
#         )
#     )

#     await session.start(
#         room=ctx.room,
#         agent=Assistant(),
#         room_input_options=RoomInputOptions(
#             # LiveKit Cloud enhanced noise cancellation
#             # - If self-hosting, omit this parameter
#             # - For telephony applications, use `BVCTelephony` for best results
#             # noise_cancellation=noise_cancellation.BVC(),
#         ),
#     )

#     await ctx.connect()

#     await session.generate_reply(
#         instructions="Greet the user and offer your assistance."
#     )


# if __name__ == "__main__":
#     agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))





# import os
# import asyncio
# import json
# import jwt
# import time
# import httpx
# from dotenv import load_dotenv
# from livekit import agents
# from livekit.agents import Agent, AgentSession, RoomInputOptions
# from livekit.plugins import openai, deepgram, elevenlabs, silero

# # Load environment
# load_dotenv()

# class VoiceAssistant(Agent):
#     def __init__(self):
#         super().__init__(
#             name="VoiceAssistant",
#             identity="ai-assistant",
#             instructions="Respond helpfully in 1-2 sentences"
#         )

# async def start_livekit_recording(room_name: str) -> dict:
#     """Start cloud recording with full error handling"""
#     try:
#         # 1. Validate environment
#         if not all([os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"), os.getenv("LIVEKIT_URL")]):
#             raise ValueError("Missing LiveKit credentials in .env")

#         # 2. Generate JWT
#         token = jwt.encode(
#             {
#                 "iss": os.getenv("LIVEKIT_API_KEY"),
#                 "exp": int(time.time()) + 600,
#                 "video": {
#                     "room_record": True,
#                     "egress": True,
#                     "room_list": True
#                 }
#             },
#             os.getenv("LIVEKIT_API_SECRET")
#         )

#         # 3. Configure HTTP client
#         async with httpx.AsyncClient() as client:
#             base_url = os.getenv("LIVEKIT_URL").replace("ws://", "http://")

#             # 4. Verify server health
#             health = await client.get(f"{base_url}/", timeout=5)
#             if health.status_code != 200:
#                 raise ConnectionError(f"Server unhealthy: {health.text}")

#             # 5. Start recording
#             response = await client.post(
#                 f"{base_url}/egress/room_composite",
#                 headers={
#                     "Authorization": f"Bearer {token}",
#                     "Content-Type": "application/json"
#                 },
#                 json={
#                     "room_name": room_name,
#                     "layout": "speaker-dark",
#                     "output": {
#                         "file": {
#                             "filepath": f"recordings/{room_name}.mp4",
#                             "file_type": "MP4"
#                         }
#                     }
#                 },
#                 timeout=30
#             )

#             # 6. Handle response
#             response.raise_for_status()
#             return response.json()

#     except json.JSONDecodeError:
#         print(f"Invalid JSON response: {response.text}")
#         return None
#     except Exception as e:
#         print(f"Recording failed: {type(e).__name__}: {str(e)}")
#         return None

# async def entrypoint(ctx: agents.JobContext):
#     # 1. Start recording
#     recording = await start_livekit_recording(ctx.room.name)
#     if not recording:
#         print("⚠️ Proceeding without recording")
#     else:
#         print(f"🔴 Recording started: {recording.get('egress_id')}")

#     # 2. Initialize agent
#     try:
#         session = AgentSession(
#             vad=silero.VAD(),
#             stt=deepgram.STT(model="nova-2"),
#             tts=elevenlabs.TTS(
#                 api_key=os.getenv("ELEVENLABS_API_KEY"),
#                 voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
#                 stability=0.7,
#                 similarity_boost=0.8
#             ),
#             llm=openai.LLM(model="gpt-4-turbo", timeout=20)
#         )
#     except Exception as e:
#         print(f"❌ Plugin init failed: {str(e)}")
#         return

#     # 3. Start session
#     await session.start(
#         room=ctx.room,
#         agent=VoiceAssistant(),
#         room_input_options=RoomInputOptions(
#             auto_subscribe=True,
#             stop_on_disconnect=True
#         )
#     )

#     # 4. Main loop
#     try:
#         await ctx.connect()
#         await session.generate_reply("Greet the user naturally.")
        
#         while ctx.room.is_connected:
#             await asyncio.sleep(1)
            
#     except Exception as e:
#         print(f"⚠️ Session error: {str(e)}")
#     finally:
#         print("🛑 Session ended")
#         if recording:
#             print("📺 Recording available in LiveKit Dashboard -> Recordings")

# if __name__ == "__main__":
#     agents.cli.run_app(
#         agents.WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             worker_server_url=os.getenv("LIVEKIT_URL")
#         )
#     )

    
# 
















# from dotenv import load_dotenv
# import os
# from livekit import agents
# from livekit.agents import AgentSession, Agent, RoomInputOptions
# from livekit.plugins import (
#     openai,
#     cartesia,
#     deepgram,
#     # noise_cancellation,
#     silero,
#     elevenlabs

# )

# import jwt
# import time
# # import requests
# import httpx

# # Load API credentials
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")



# # from livekit.plugins.turn_detector.multilingual import MultilingualModel

# load_dotenv(dotenv_path=".env.local")

# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(instructions="You are a helpful voice AI assistant.")

# def create_server_token():
#     now = int(time.time())
#     payload = {
#         "iss": LIVEKIT_API_KEY,
#         "exp": now + 600,
#         "jti": str(now),
#         "video": {
#             "room_create": True,
#             "room_list": True,
#             "egress": True
#         }
#     }
#     return jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")

# async def start_egress_httpx(room_name: str, output_file_path: str):
#     token = create_server_token()

#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "room_name": room_name,
#         "layout": "grid",
#         "output": {
#             "file": {
#                 "filepath": output_file_path
#             }
#         }
#     }

#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             f"{LIVEKIT_URL}/egress/composite",
#             headers=headers,
#             json=payload
#         )

#         print("Egress started:", response.status_code)

#         try:
#             data = response.json()
#             print("Response JSON:", data)
#             return data
#         except Exception:
#             print("Raw response text:", response.text)
#             return None




# async def entrypoint(ctx: agents.JobContext):
#     session = AgentSession(
#         # stt=deepgram.STT(model="nova-3", language="multi"),
#         # llm=openai.LLM(model="gpt-4o-mini"),
#         # tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
#         vad=silero.VAD.load(),
#         # turn_detection=MultilingualModel(),
#         llm=openai.LLM(model="llama-3.3-70b-versatile"),
#         stt=deepgram.STT(model="nova-3"),
#         tts=elevenlabs.TTS(  # Changed from cartesia to elevenlabs
#             api_key=ELEVENLABS_API_KEY,
#             voice_id=ELEVENLABS_VOICE_ID,
#             # model="eleven_monolingual_v2",
#             # stability=0.5,
#             # similarity_boost=0.75
#         ),
#     )

#     await session.start(
#         room=ctx.room,
#         agent=Assistant(),
#         room_input_options=RoomInputOptions(
#             # LiveKit Cloud enhanced noise cancellation
#             # - If self-hosting, omit this parameter
#             # - For telephony applications, use `BVCTelephony` for best results
#             # noise_cancellation=noise_cancellation.BVC(), 
#         ),
#     )

#     # await start_egress_httpx(room_name=ctx.room.name, output_file_path="recordings/session1.mp4")

#     await ctx.connect()

#     await session.generate_reply(
#         instructions="Greet the user and offer your assistance."
#     )


# if __name__ == "__main__":
#     agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))






from dotenv import load_dotenv
import jwt
import time
import httpx
import os
import os
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    # deepgram,
    # noise_cancellation,
    silero,
    elevenlabs
)
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

import deepgram

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")



# from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(dotenv_path=".env.local")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant. Keep the responses short not more than 2 sentences")



async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        # stt=deepgram.STT(model="nova-3", language="multi"),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),
        llm=openai.LLM(model="llama-3.3-70b-versatile"), #groq
        stt=deepgram.STT(model="nova-3"),
        tts=elevenlabs.TTS(  # Changed from cartesia to elevenlabs
            api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            # model="eleven_monolingual_v2",
            # stability=0.5,
            # similarity_boost=0.75
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            # noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))











# import os
# import asyncio
# import numpy as np
# import sounddevice as sd
# from dotenv import load_dotenv
# from livekit.plugins import(
#     deepgram
# )
# # from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
# from livekit import agents
# from livekit.agents import Agent, AgentSession
# from livekit.plugins import elevenlabs

# # Load env variables
# load_dotenv(dotenv_path=".env.local")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# # Audio config
# SAMPLE_RATE = 16000
# CHANNELS = 1

# # Custom Agent that handles text
# class Assistant(Agent):
#     async def on_text(self, text: str) -> str:
#         print(f"User: {text}")
#         return await self.generate_reply(text)

# # Entrypoint for LiveKit agent session
# async def entrypoint(ctx: agents.JobContext):
#     session = AgentSession(
#         llm=agents.openai.LLM(model="gpt-4o-mini"),  # Or any other
#         stt=None,  # No built-in STT
#         tts=elevenlabs.TTS(
#             api_key=ELEVENLABS_API_KEY,
#             voice_id=ELEVENLABS_VOICE_ID
#         ),
#     )

#     await session.start(room=ctx.room, agent=Assistant())
#     await ctx.connect()

#     # Run Deepgram STT in parallel
#     asyncio.create_task(run_deepgram_stt(session))

#     await session.generate_reply(instructions="Hi! I’m your assistant. How can I help you today?")

# # Gating logic for transcripts
# def gate_passes(transcript: str) -> bool:
#     words = transcript.strip().split()
#     return len(words) >= 3  # Threshold: minimum 3 words

# # Deepgram streaming and LiveKit integration
# async def run_deepgram_stt(session: AgentSession):
#     deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
#     dg_conn = deepgram.listen.v("1")

#     # Handler for received transcript
#     def on_message(self, result, **kwargs):
#         if not result.is_final:
#             return
#         transcript = result.channel.alternatives[0].transcript
#         if transcript and gate_passes(transcript):
#             print(f"✅ Passed: {transcript}")
#             asyncio.run_coroutine_threadsafe(session.process_text(transcript), asyncio.get_event_loop())
#         else:
#             print(f"❌ Rejected: {transcript}")

#     def on_error(error, **kwargs):
#         print(f"[Deepgram ERROR] {error}")

#     dg_conn.on(LiveTranscriptionEvents.Transcript, on_message)
#     dg_conn.on(LiveTranscriptionEvents.Error, on_error)

#     options = LiveOptions(
#         model="nova-3",
#         encoding="linear16",
#         sample_rate=SAMPLE_RATE,
#         channels=CHANNELS,
#         interim_results=True
#     )

#     if not dg_conn.start(options):
#         print("Failed to start Deepgram connection")
#         return

#     print("🎙️ Start speaking! Press Ctrl+C to stop.")

#     # Microphone stream
#     try:
#         with sd.InputStream(
#             samplerate=SAMPLE_RATE,
#             channels=CHANNELS,
#             dtype='float32',
#             callback=lambda indata, *_: dg_conn.send((indata * 32767).astype(np.int16).tobytes())
#         ):
#             while True:
#                 await asyncio.sleep(0.1)
#     except KeyboardInterrupt:
#         print("\n🔴 Stopping...")
#     finally:
#         dg_conn.finish()

# # CLI runner
# if __name__ == "__main__":
#     agents.cli.run_app(
#         agents.WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             api_key=LIVEKIT_API_KEY,
#             api_secret=LIVEKIT_API_SECRET,
#             url=LIVEKIT_URL
#         )
#     )















# import logging
# from dataclasses import dataclass, field
# from typing import Optional
# import os

# from dotenv import load_dotenv

# from livekit import api
# from livekit.agents import (
#     Agent,
#     AgentSession,
#     ChatContext,
#     JobContext,
#     JobProcess,
#     RoomInputOptions,
#     RoomOutputOptions,
#     RunContext,
#     WorkerOptions,
#     cli,
#     metrics,
# )
# from livekit.agents.job import get_job_context
# from livekit.agents.llm import function_tool
# from livekit.agents.voice import MetricsCollectedEvent
# from livekit.plugins import deepgram, openai, silero, elevenlabs  # Changed from cartesia to elevenlabs

# logger = logging.getLogger("multi-agent")

# # Load environment variables
# load_dotenv(dotenv_path=".env.local")

# # Get ElevenLabs API key from environment
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Default voice if not specified

# common_instructions = (
#     "You are an editor at a leading publishing house, with a strong track record "
#     "of discovering and nurturing new talent. You are a great communicator and ask "
#     "the right questions to get the best out of people. You want the best for your "
#     "authors, and you are not afraid to tell them when their ideas are not good enough."
# )

# @dataclass
# class CharacterData:
#     name: Optional[str] = None
#     background: Optional[str] = None

# @dataclass
# class StoryData:
#     characters: list[CharacterData] = field(default_factory=list)
#     locations: list[str] = field(default_factory=list) 
#     theme: Optional[str] = None

# class LeadEditorAgent(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=f"{common_instructions} You are the lead editor at this business, "
#             "and are yourself a generalist -- but employ several specialist editors, "
#             "specializing in childrens' books and fiction, respectively. You trust your "
#             "editors to do their jobs, and will hand off the conversation to them when you feel "
#             "you have an idea of the right one."
#             "Your goal is to gather a few pieces of information from the user about their next"
#             "idea for a short story, and then hand off to the right agent."
#             "Start the conversation with a short introduction, then get straight to the "
#             "details. You may hand off to either editor as soon as you know which one is the right fit.",
#         )

#     async def on_enter(self):
#         self.session.generate_reply()

#     @function_tool
#     async def character_introduction(self, context: RunContext[StoryData], name: str, background: str):
#         character = CharacterData(name=name, background=background)
#         context.userdata.characters.append(character)
#         logger.info("added character to the story: %s", name)

#     @function_tool
#     async def location_introduction(self, context: RunContext[StoryData], location: str):
#         context.userdata.locations.append(location)
#         logger.info("added location to the story: %s", location)

#     @function_tool
#     async def theme_introduction(self, context: RunContext[StoryData], theme: str):
#         context.userdata.theme = theme
#         logger.info("set theme to the story: %s", theme)

#     @function_tool
#     async def detected_childrens_book(self, context: RunContext[StoryData]):
#         childrens_editor = SpecialistEditorAgent("children's books", chat_ctx=context.session._chat_ctx)
#         logger.info("switching to the children's book editor with the provided user data: %s", context.userdata)
#         return childrens_editor, "Let's switch to the children's book editor."

#     @function_tool
#     async def detected_novel(self, context: RunContext[StoryData]):
#         childrens_editor = SpecialistEditorAgent("novels", chat_ctx=context.session._chat_ctx)
#         logger.info("switching to the children's book editor with the provided user data: %s", context.userdata)
#         return childrens_editor, "Let's switch to the children's book editor."

# class SpecialistEditorAgent(Agent):
#     def __init__(self, specialty: str, chat_ctx: Optional[ChatContext] = None) -> None:
#         super().__init__(
#             instructions=f"{common_instructions}. You specialize in {specialty}, and have "
#             "worked with some of the greats, and have even written a few books yourself.",
#             tts=elevenlabs.TTS(  # Changed from openai.TTS to elevenlabs.TTS
#                 api_key=ELEVENLABS_API_KEY,
#                 voice_id=ELEVENLABS_VOICE_ID,
#                 model="eleven_monolingual_v2",
#                 stability=0.5,
#                 similarity_boost=0.75
#             ),
#             chat_ctx=chat_ctx,
#         )

#     async def on_enter(self):
#         self.session.generate_reply()

#     @function_tool
#     async def character_introduction(self, context: RunContext[StoryData], name: str, background: str):
#         character = CharacterData(name=name, background=background)
#         context.userdata.characters.append(character)
#         logger.info("added character to the story: %s", name)

#     @function_tool
#     async def location_introduction(self, context: RunContext[StoryData], location: str):
#         context.userdata.locations.append(location)
#         logger.info("added location to the story: %s", location)

#     @function_tool
#     async def theme_introduction(self, context: RunContext[StoryData], theme: str):
#         context.userdata.theme = theme
#         logger.info("set theme to the story: %s", theme)

#     @function_tool
#     async def story_finished(self, context: RunContext[StoryData]):
#         self.session.interrupt()
#         await self.session.generate_reply(
#             instructions="give brief but honest feedback on the story idea", 
#             allow_interruptions=False
#         )
#         job_ctx = get_job_context()
#         await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()

# async def entrypoint(ctx: JobContext):
#     await ctx.connect()

#     session = AgentSession[StoryData](
#         vad=ctx.proc.userdata["vad"],
#         llm=openai.LLM(model="llama-3.3-70b-versatile"),
#         stt=deepgram.STT(model="nova-3"),
#         tts=elevenlabs.TTS(  # Changed from cartesia to elevenlabs
#             api_key=ELEVENLABS_API_KEY,
#             voice_id=ELEVENLABS_VOICE_ID,
#             # model="eleven_monolingual_v2",
#             # stability=0.5,
#             # similarity_boost=0.75
#         ),
#         userdata=StoryData(),
#     )

#     usage_collector = metrics.UsageCollector()

#     @session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     async def log_usage():
#         summary = usage_collector.get_summary()
#         logger.info(f"Usage: {summary}")

#     ctx.add_shutdown_callback(log_usage)

#     await session.start(
#         agent=LeadEditorAgent(),
#         room=ctx.room,
#         room_input_options=RoomInputOptions(),
#         room_output_options=RoomOutputOptions(transcription_enabled=True),
#     )

# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))



# import logging
# from dataclasses import dataclass, field
# from typing import Optional

# from dotenv import load_dotenv

# from livekit import api
# from livekit.agents import (
#     Agent,
#     AgentSession,
#     ChatContext,
#     JobContext,
#     JobProcess,
#     RoomInputOptions,
#     RoomOutputOptions,
#     RunContext,
#     WorkerOptions,
#     cli,
#     metrics,
# )
# from livekit.agents.job import get_job_context
# from livekit.agents.llm import function_tool
# from livekit.agents.voice import MetricsCollectedEvent
# from livekit.plugins import deepgram, openai, silero, cartesia

# # (print(cartesia.TTS))

# # uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
# # from livekit.plugins import noise_cancellation

# ## The storyteller agent is a multi-agent that can handoff the session to another agent.
# ## This example demonstrates more complex workflows with multiple agents.
# ## Each agent could have its own instructions, as well as different STT, LLM, TTS,
# ## or realtime models.

# logger = logging.getLogger("multi-agent")

# load_dotenv(dotenv_path=".env.local")

# common_instructions = (
#     "You are an editor at a leading publishing house, with a strong track record "
#     "of discovering and nurturing new talent. You are a great communicator and ask "
#     "the right questions to get the best out of people. You want the best for your "
#     "authors, and you are not afraid to tell them when their ideas are not good enough."
# )


# @dataclass
# class CharacterData:
#     # Shared data that's used by the editor agent.
#     # This structure is passed as a parameter to function calls.

#     name: Optional[str] = None
#     background: Optional[str] = None


# @dataclass
# class StoryData:
#     # Shared data that's used by the editor agent.
#     # This structure is passed as a parameter to function calls.

#     characters: list[CharacterData] = field(default_factory=list)
#     locations: list[str] = field(default_factory=list) 
#     theme: Optional[str] = None


# class LeadEditorAgent(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=f"{common_instructions} You are the lead editor at this business, "
#             "and are yourself a generalist -- but empoly several specialist editors, "
#             "specializing in childrens' books and fiction, respectively. You trust your "
#             "editors to do their jobs, and will hand off the conversation to them when you feel "
#             "you have an idea of the right one."
#             "Your goal is to gather a few pieces of information from the user about their next"
#             "idea for a short story, and then hand off to the right agent."
#             "Start the conversation with a short introduction, then get straight to the "
#             "details. You may hand off to either editor as soon as you know which one is the right fit.",
#         )

#     async def on_enter(self):
#         # when the agent is added to the session, it'll generate a reply
#         # according to its instructions
#         self.session.generate_reply()

#     @function_tool
#     async def character_introduction(
#         self,
#         context: RunContext[StoryData],
#         name: str,
#         background: str,
#     ):
#         """Called when the user has provided a character.

#         Args:
#             name: The name of the character
#             background: The character's history, occupation, and other details
#         """

#         character = CharacterData(name=name, background=background)
#         context.userdata.characters.append(character)

#         logger.info(
#             "added character to the story: %s", name
#         )

#     @function_tool
#     async def location_introduction(
#         self,
#         context: RunContext[StoryData],
#         location: str,
#     ):
#         """Called when the user has provided a location.

#         Args:
#             location: The name of the location
#         """

#         context.userdata.locations.append(location)

#         logger.info(
#             "added location to the story: %s", location
#         )

#     @function_tool
#     async def theme_introduction(
#         self,
#         context: RunContext[StoryData],
#         theme: str,
#     ):
#         """Called when the user has provided a theme.

#         Args:
#             theme: The name of the theme
#         """

#         context.userdata.theme = theme

#         logger.info(
#             "set theme to the story: %s", theme
#         )

#     @function_tool
#     async def detected_childrens_book(
#         self,
#         context: RunContext[StoryData],
#     ):
#         """Called when the user has provided enough information to suggest a children's book.
#         """

#         childrens_editor = SpecialistEditorAgent("children's books", chat_ctx=context.session._chat_ctx)
#         # here we are creating a ChilrensEditorAgent with the full chat history,
#         # as if they were there in the room with the user the whole time.
#         # we could also omit it and rely on the userdata to share context.

#         logger.info(
#             "switching to the children's book editor with the provided user data: %s", context.userdata
#         )
#         return childrens_editor, "Let's switch to the children's book editor."

#     @function_tool
#     async def detected_novel(
#         self,
#         context: RunContext[StoryData],
#     ):
#         """Called when the user has provided enough information to suggest a children's book.
#         """

#         childrens_editor = SpecialistEditorAgent("novels", chat_ctx=context.session._chat_ctx)
#         # here we are creating a ChilrensEditorAgent with the full chat history,
#         # as if they were there in the room with the user the whole time.
#         # we could also omit it and rely on the userdata to share context.

#         logger.info(
#             "switching to the children's book editor with the provided user data: %s", context.userdata
#         )
#         return childrens_editor, "Let's switch to the children's book editor."


# class SpecialistEditorAgent(Agent):
#     def __init__(self, specialty: str, chat_ctx: Optional[ChatContext] = None) -> None:
#         super().__init__(
#             instructions=f"{common_instructions}. You specialize in {specialty}, and have "
#             "worked with some of the greats, and have even written a few books yourself.",
#             # each agent could override any of the model services, including mixing
#             # realtime and non-realtime models
#             tts=openai.TTS(voice="echo"),
#             chat_ctx=chat_ctx,
#         )

#     async def on_enter(self):
#         # when the agent is added to the session, we'll initiate the conversation by
#         # using the LLM to generate a reply
#         self.session.generate_reply()

#     @function_tool
#     async def character_introduction(
#         self,
#         context: RunContext[StoryData],
#         name: str,
#         background: str,
#     ):
#         """Called when the user has provided a character.

#         Args:
#             name: The name of the character
#             background: The character's history, occupation, and other details
#         """

#         character = CharacterData(name=name, background=background)
#         context.userdata.characters.append(character)

#         logger.info(
#             "added character to the story: %s", name
#         )

#     @function_tool
#     async def location_introduction(
#         self,
#         context: RunContext[StoryData],
#         location: str,
#     ):
#         """Called when the user has provided a location.

#         Args:
#             location: The name of the location
#         """

#         context.userdata.locations.append(location)

#         logger.info(
#             "added location to the story: %s", location
#         )

#     @function_tool
#     async def theme_introduction(
#         self,
#         context: RunContext[StoryData],
#         theme: str,
#     ):
#         """Called when the user has provided a theme.

#         Args:
#             theme: The name of the theme
#         """

#         context.userdata.theme = theme

#         logger.info(
#             "set theme to the story: %s", theme
#         )

#     @function_tool
#     async def story_finished(self, context: RunContext[StoryData]):
#         """When the editor think the broad strokes of the story have been hammered out,
#         they can stop you with their final thoughts.
#         """
#         # interrupt any existing generation
#         self.session.interrupt()

#         # generate a goodbye message and hang up
#         # awaiting it will ensure the message is played out before returning
#         await self.session.generate_reply(
#             instructions="give brief but honest feedback on the story idea", allow_interruptions=False
#         )

#         job_ctx = get_job_context()
#         await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()


# async def entrypoint(ctx: JobContext):
#     await ctx.connect()

#     session = AgentSession[StoryData](
#         vad=ctx.proc.userdata["vad"],
#         # any combination of STT, LLM, TTS, or realtime API can be used
#         llm=openai.LLM(model="llama-3.3-70b-versatile"),
#         stt=deepgram.STT(model="nova-3"),
#         # tts=openai.TTS(voice="ash"),
#         tts=cartesia.TTS(
#         model="sonic-2",  # Cartesia supports "sonic-2" and others
#         voice="default",  # You can later change to a voice UUID if you want custom voices
#         speed=1.0,
#         emotion=["neutral"],  # Can be "neutral", "happy", etc.
#         # streamed = False,
#          ),
#         userdata=StoryData(),
#     )

#     # log metrics as they are emitted, and total usage after session is over
#     usage_collector = metrics.UsageCollector()

#     @session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     async def log_usage():
#         summary = usage_collector.get_summary()
#         logger.info(f"Usage: {summary}")

#     ctx.add_shutdown_callback(log_usage)

#     await session.start(
#         agent=LeadEditorAgent(),
#         room=ctx.room,
#         room_input_options=RoomInputOptions(
#             # uncomment to enable Krisp BVC noise cancellation
#             # noise_cancellation=noise_cancellation.BVC(),
#         ),
#         room_output_options=RoomOutputOptions(transcription_enabled=True),
#     )


# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
