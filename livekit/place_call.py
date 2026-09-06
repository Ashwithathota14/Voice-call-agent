"""Place an outbound call on an Indian number through a RingTrunk trunk.

    LIVEKIT_URL=wss://<project>.livekit.cloud LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \
    OUTBOUND_TRUNK_ID=ST_xxx CALLER_ID=9XXXXXXXXX \
    python place_call.py +91XXXXXXXXXX [room-name]

CALLER_ID must be one of the numbers attached to your RingTrunk trunk, or the call
is rejected with SIP 403 cli_not_allowed. Destinations are Indian mobile numbers
only: 10 digits starting 6, 7, 8 or 9.
"""
import asyncio
import os
import sys

from livekit import api


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: place_call.py +91XXXXXXXXXX [room-name]")
    destination = sys.argv[1]
    room_name = sys.argv[2] if len(sys.argv) > 2 else "call-outbound"

    lkapi = api.LiveKitAPI()  # reads LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
    try:
        participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=os.environ["OUTBOUND_TRUNK_ID"],
                sip_call_to=destination,
                sip_number=os.environ["CALLER_ID"],
                room_name=room_name,
                participant_identity="phone",
                # Wait for the callee to pick up before the call is considered live.
                wait_until_answered=True,
            )
        )
        print(f"dialled {destination} into room {room_name}: {participant.participant_id}")
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(main())
