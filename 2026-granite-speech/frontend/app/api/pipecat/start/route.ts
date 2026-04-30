import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.PIPECAT_BACKEND_URL || 'http://127.0.0.1:7860';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_URL}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
    console.error('Pipecat start proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to connect to voice backend' },
      { status: 502 }
    );
  }
}
