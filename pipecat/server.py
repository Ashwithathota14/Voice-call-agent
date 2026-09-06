"""Starts a Pipecat bot when a phone call lands in a LiveKit room.

LiveKit sends a `participant_joined` webhook when the SIP participant (the
caller) joins the room the dispatch rule created. We verify the webhook with
the LiveKit API key/secret and start bot.py for that room.

    uvicorn server:app --port 8080

Set your LiveKit project's webhook URL to http://<host>:8080/livekit-webhook.
"""
import asyncio
import os

from fastapi import FastAPI, Header, HTTPException, Request
from livekit.api import TokenVerifier, WebhookReceiver

from bot import run_bot

app = FastAPI()
receiver = WebhookReceiver(
    TokenVerifier(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
)
running: dict[str, asyncio.Task] = {}


@app.post("/livekit-webhook")
async def livekit_webhook(request: Request, authorization: str = Header(default="")):
    body = (await request.body()).decode()
    try:
        event = receiver.receive(body, authorization)
    except Exception as exc:  # bad signature or malformed event
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if event.event != "participant_joined":
        return {"ok": True}

    participant = event.participant
    room = event.room.name
    # Only the phone leg should trigger a bot; the bot itself joins as "agent".
    is_phone = participant.kind == 3 or participant.identity.startswith("sip_")  # 3 == SIP
    if not is_phone or room in running:
        return {"ok": True}

    task = asyncio.create_task(run_bot(room))
    running[room] = task
    task.add_done_callback(lambda _t: running.pop(room, None))
    return {"ok": True, "started": room}
