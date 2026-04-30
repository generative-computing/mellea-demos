import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.PIPECAT_BACKEND_URL || 'http://127.0.0.1:7860';

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  try {
    const { sessionId } = await params;
    const body = await request.text();

    const response = await fetch(
      `${BACKEND_URL}/sessions/${sessionId}/api/offer`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      }
    );

    if (!response.ok) {
      const text = await response.text();
      return NextResponse.json(
        { error: `Backend returned ${response.status}: ${text}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Pipecat offer proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to negotiate WebRTC offer' },
      { status: 502 }
    );
  }
}
