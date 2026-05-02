import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.PIPECAT_BACKEND_URL || 'http://127.0.0.1:7860';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/ivr/config`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

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
    console.error('IVR config proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch IVR config' },
      { status: 502 }
    );
  }
}
