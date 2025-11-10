import { NextResponse } from "next/server";
import { AccessToken, type AccessTokenOptions, type VideoGrant } from "livekit-server-sdk";

const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const LIVEKIT_URL = process.env.NEXT_PUBLIC_LIVEKIT_URL;


export type ConnectionDetails = {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
  // Include personalization data in response
  name?: string;
  age?: string;
  interests?: string[];
};

export async function GET(request: Request) {
  try {
    if (!LIVEKIT_URL || !API_KEY || !API_SECRET) {
      throw new Error("Missing LiveKit environment variables");
    }

    const url = new URL(request.url);
    const name = url.searchParams.get("name") || "user";
    const age = url.searchParams.get("age") || "";
    const interestsString = url.searchParams.get("interests") || "";
    const interests = interestsString ? interestsString.split(",") : [];

    const participantIdentity = `voice_assistant_user_${Math.floor(Math.random() * 10_000)}`;
    const roomName = `voice_assistant_room_${Math.floor(Math.random() * 10_000)}`;

    const participantToken = await createParticipantToken(
      { identity: participantIdentity, name },
      roomName
    );

    // Prepare connection details including personalization info
    const data: ConnectionDetails = {
      serverUrl: LIVEKIT_URL,
      roomName,
      participantName: name,
      participantToken,
      name,
      age,
      interests,
    };

    // Optionally pass personalization to your Python agent backend here
    await notifyAgentBackend(data);

    return NextResponse.json(data, {
      headers: {
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*", // Or set your frontend origin for more security
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);
      return new NextResponse(error.message, { status: 500 });
    }
  }
}

// Generate LiveKit participant token
function createParticipantToken(userInfo: AccessTokenOptions, roomName: string) {
  const at = new AccessToken(API_KEY!, API_SECRET!, {
    ...userInfo,
    ttl: "15m",
  });

  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  };
  at.addGrant(grant);
  return at.toJwt();
}

// Notify the Python agent backend with personalization (optional, adjust URL & payload)
async function notifyAgentBackend(connectionDetails: ConnectionDetails) {
  try {
    await fetch("http://your-python-agent-backend/agent-personalization", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        room: connectionDetails.roomName,
        participant: connectionDetails.participantName,
        age: connectionDetails.age,
        interests: connectionDetails.interests,
      }),
    });
  } catch (err) {
    console.error("Failed to notify agent backend:", err);
  }
}
