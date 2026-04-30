'use client';

import dynamic from 'next/dynamic';
import Image from 'next/image';

const GraniteSpeechDemo = dynamic(() => import('./components/GraniteSpeechDemo'), {
  ssr: false,
});

export default function SpeechDemoPage() {
  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#121619',
      }}
    >
      {/* Top bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 32px',
          flexShrink: 0,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <Image
            src="/assets/granite-speech-logo.png"
            alt="Granite Speech + Mellea"
            width={280}
            height={55}
            unoptimized
          />
        </div>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <a
            href="https://huggingface.co/collections/ibm-granite/granite-speech"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px',
              border: '1px solid #2f477d',
              borderRadius: '8px',
              width: '124px',
              height: '31px',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              textDecoration: 'none',
            }}
          >
            <Image
              src="/assets/hf-logo.svg"
              alt="HuggingFace"
              width={16}
              height={15}
              unoptimized
            />
            <span
              style={{
                fontSize: '14px',
                lineHeight: '20px',
                color: '#ffffff',
                letterSpacing: '0.16px',
                whiteSpace: 'nowrap',
              }}
            >
              HuggingFace
            </span>
          </a>

          <a
            href="https://github.com/generative-computing"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px',
              border: '1px solid #3e4e73',
              borderRadius: '8px',
              width: '82px',
              height: '31px',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              textDecoration: 'none',
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="white"
              aria-hidden="true"
            >
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            <span
              style={{
                fontSize: '14px',
                lineHeight: '20px',
                color: '#ffffff',
                letterSpacing: '0.16px',
                whiteSpace: 'nowrap',
              }}
            >
              GitHub
            </span>
          </a>
        </div>
      </div>

      {/* Voice demo — fills remaining space */}
      <div style={{ flex: 1, padding: '0 32px 32px', minHeight: 0 }}>
        <GraniteSpeechDemo />
      </div>
    </div>
  );
}
