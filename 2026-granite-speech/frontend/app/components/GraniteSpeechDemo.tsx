'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import '@pipecat-ai/voice-ui-kit/styles.scoped.css';
import {
  PipecatAppBase,
  Conversation,
  ConnectButton,
  SpinLoader,
  usePipecatConnectionState,
  usePipecatConversation,
} from '@pipecat-ai/voice-ui-kit';
import { useRTVIClientEvent } from '@pipecat-ai/client-react';
import { RTVIEvent } from '@pipecat-ai/client-js';
import { Toggle } from '@carbon/react';

type CellState = 'pending' | 'pass' | 'fail';

interface ValidationTurn {
  turnId: string;
  requirements: string[];
  nSamples: number;
  // cells[gen][req] -> CellState
  cells: CellState[][];
  sampleStarted: boolean[];
  texts: (string | null)[];
  chosenGen: number | null;
  elapsedMs: number | null;
  createdAt: number;
}

function makeTurn(turnId: string, nSamples: number, requirements: string[]): ValidationTurn {
  return {
    turnId,
    requirements,
    nSamples,
    cells: Array.from({ length: nSamples }, () =>
      Array.from({ length: requirements.length }, () => 'pending' as CellState),
    ),
    sampleStarted: Array.from({ length: nSamples }, () => false),
    texts: Array.from({ length: nSamples }, () => null),
    chosenGen: null,
    elapsedMs: null,
    createdAt: Date.now(),
  };
}

const INTRINSICS = [
  { id: 'intrinsic1', label: 'Use requirement_checker', enabled: true },
];

export default function GraniteSpeechDemo() {
  const [activeIntrinsic, setActiveIntrinsic] = useState<string | null>(null);
  
  const ivrValidationRef = useRef(activeIntrinsic === 'intrinsic1');
  ivrValidationRef.current = activeIntrinsic === 'intrinsic1';

  const transportOptions = useMemo(() => ({
    offerUrlTemplate: '/api/offer/:sessionId',
  }), []);

  const startBotParams = useMemo(() => ({
    endpoint: '/api/pipecat/start',
    get requestData() {
      return {
        createDailyRoom: false,
        enableDefaultIceServers: true,
        transport: 'webrtc',
        ivrValidation: ivrValidationRef.current,
      };
    }
  }), []);

  return (
    <div
      className="dark"
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '8px',
        // @ts-expect-error CSS custom properties
        '--color-background': '#121619',
        '--color-card': '#141c2e',
        '--color-border': 'rgba(38, 54, 89, 0.8)',
        '--color-primary': '#32a6ff',
        '--font-sans': "'IBM Plex Sans', sans-serif",
      }}
    >
      <PipecatAppBase
        transportType="smallwebrtc"
        transportOptions={transportOptions}
        startBotParams={startBotParams}
        noThemeProvider
      >
        {({ client, handleConnect, handleDisconnect }) =>
          client ? (
            <IntrinsicSelector
              client={client}
              activeIntrinsic={activeIntrinsic}
              onIntrinsicChange={setActiveIntrinsic}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
            />
          ) : null
        }
      </PipecatAppBase>
    </div>
  );
}

function IntrinsicSelector({
  client,
  activeIntrinsic,
  onIntrinsicChange,
  onConnect,
  onDisconnect,
}: {
  client: any;
  activeIntrinsic: string | null;
  onIntrinsicChange: (id: string | null) => void;
  onConnect?: () => void | Promise<void>;
  onDisconnect?: () => void | Promise<void>;
}) {
  const { isConnected, isConnecting, isDisconnected } = usePipecatConnectionState();
  const { messages } = usePipecatConversation();
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  useRTVIClientEvent(RTVIEvent.BotStartedSpeaking, () => setIsBotSpeaking(true));
  useRTVIClientEvent(RTVIEvent.BotStoppedSpeaking, () => setIsBotSpeaking(false));

  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [validationTurns, setValidationTurns] = useState<ValidationTurn[]>([]);
  const wasConnectedRef = useRef(false);
  const hadMessagesRef = useRef(false);

  const [panelCollapsed, setPanelCollapsed] = useState(true);
  const [ivrConfig, setIvrConfig] = useState<{ requirements: string[]; nSamples: number } | null>(null);

  useEffect(() => {
    if (ivrConfig) return;
    let cancelled = false;
    fetch('/api/ivr/config')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((data) => {
        if (cancelled) return;
        if (Array.isArray(data?.requirements) && typeof data?.nSamples === 'number') {
          setIvrConfig({ requirements: data.requirements, nSamples: data.nSamples });
        }
      })
      .catch((err) => console.error('Failed to load IVR config', err));
    return () => {
      cancelled = true;
    };
  }, [ivrConfig]);

  useEffect(() => {
    if (!client) return;
    const handleServerMessage = (data: unknown) => {
      const payload = (data && typeof data === 'object' && 'data' in (data as Record<string, unknown>))
        ? ((data as { data: unknown }).data as Record<string, unknown>)
        : (data as Record<string, unknown>);
      if (!payload || typeof payload !== 'object') return;
      if (payload.type !== 'ivr_validation') return;

      const turnId = payload.turn_id as string;
      const phase = payload.phase as string;

      setValidationTurns((prev) => {
        const idx = prev.findIndex((t) => t.turnId === turnId);
        const next = prev.slice();

        if (phase === 'start') {
          const turn = makeTurn(
            turnId,
            (payload.n_samples as number) ?? 3,
            (payload.requirements as string[]) ?? [],
          );
          if (idx >= 0) next[idx] = turn;
          else next.unshift(turn);
          return next.slice(0, 3);
        }

        if (idx < 0) return prev;
        const turn = { ...next[idx] };

        if (phase === 'sample_start') {
          const gen = payload.gen as number;
          turn.sampleStarted = turn.sampleStarted.slice();
          turn.sampleStarted[gen] = true;
        } else if (phase === 'sample_text') {
          const gen = payload.gen as number;
          const text = payload.text as string;
          turn.texts = turn.texts.slice();
          turn.texts[gen] = text;
        } else if (phase === 'check') {
          const gen = payload.gen as number;
          const reqIndex = payload.req_index as number;
          const passed = payload.passed as boolean;
          turn.cells = turn.cells.map((row) => row.slice());
          if (turn.cells[gen]) {
            turn.cells[gen][reqIndex] = passed ? 'pass' : 'fail';
          }
        } else if (phase === 'done') {
          turn.chosenGen = payload.chosen_gen as number;
          turn.elapsedMs = payload.elapsed_ms as number;
        }

        next[idx] = turn;
        return next;
      });
    };

    // Subscribe via the client directly. The pipecat client is an EventEmitter;
    // 'serverMessage' is the RTVIEvent name for RTVIServerMessageFrame deliveries.
    client.on?.('serverMessage', handleServerMessage);
    return () => {
      client.off?.('serverMessage', handleServerMessage);
    };
  }, [client]);

  useEffect(() => {
    if (isConnected && !wasConnectedRef.current) {
      setStatusMessage(null);
      setValidationTurns([]);
    }
    if (wasConnectedRef.current && isDisconnected) {
      if (hadMessagesRef.current) {
        setStatusMessage('Not connected to an agent. Click the connect button to start a new conversation.');
      }
    }
    wasConnectedRef.current = isConnected;
  }, [isConnected, isDisconnected]);

  const hasMessages = messages.length > 0;
  hadMessagesRef.current = hasMessages;
  const showEmptyState = isDisconnected && !hasMessages;
  const showConnectingState = isConnecting;
  const lastMessage = messages[messages.length - 1];
  const agentIdle =
    isConnected &&
    !isBotSpeaking &&
    lastMessage?.role === 'assistant' &&
    lastMessage?.final === true;
  const showWaitingForMessages = isConnected && !isBotSpeaking && (!hasMessages || agentIdle);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: 'calc(100vh - 120px)', width: '100%', maxWidth: '900px', margin: '0 auto' }}>
      {/* Chat row: chat panel + optional validation side panel */}
      <div style={{ flex: 1, display: 'flex', gap: '12px', minHeight: 0, position: 'relative' }}>
      {/* Chat panel */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#141c2e',
          border: '1px solid #263659',
          borderRadius: '8px',
          overflow: 'hidden',
          minHeight: 0,
          position: 'relative',
          zIndex: 1,
        }}
      >
        {showEmptyState && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '32px',
              textAlign: 'center',
              gap: '8px',
              zIndex: 2,
            }}
          >
            <div style={{ color: '#99bbd5', fontSize: '16px', lineHeight: '24px' }}>
              Not Connected to an Agent
            </div>
            <div style={{ color: '#ffffff', fontSize: '14px', lineHeight: '20px' }}>
              Click on the connect button to connect to an agent and see conversation messages in real-time.
            </div>
            <div style={{ marginTop: '16px' }}>
              <ConnectButton
                onConnect={onConnect}
                onDisconnect={onDisconnect}
                stateContent={{
                  disconnected: { children: 'Connect', variant: 'active' },
                  ready: { children: 'Disconnect', variant: 'destructive' },
                }}
                size="xl"
                className="connect-btn-full"
              />
            </div>
          </div>
        )}

        <div
          className="transcript-wrapper"
          style={{
            visibility:
              showEmptyState || showConnectingState || !hasMessages ? 'hidden' : 'visible',
          }}
        >
          <Conversation
            assistantLabel="Agent"
            clientLabel="User"
            noTextInput
            noFunctionCalls
            classNames={{ message: 'transcript-messages', time: 'transcript-hide' }}
          />
        </div>

        <div
          style={{
            flexShrink: 0,
            padding: '0 8px 0 16px',
            borderTop: '1px solid #263659',
            backgroundColor: '#1c2435',
            color: '#99bbd5',
            fontSize: '12px',
            lineHeight: '18px',
            height: '30px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            letterSpacing: '0.16px',
            zIndex: 3,
          }}
          aria-live="polite"
          role="status"
        >
          <span
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {showConnectingState && <SpinLoader size={14} />}
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {showConnectingState
                ? 'Please wait one moment while we connect you with an agent.'
                : showWaitingForMessages
                  ? 'Waiting for messages…'
                  : statusMessage ?? ''}
            </span>
          </span>
          {isConnected && (
            <ConnectButton
              onConnect={onConnect}
              onDisconnect={onDisconnect}
              stateContent={{
                disconnected: { children: 'Connect', variant: 'active' },
                ready: { children: 'Disconnect', variant: 'destructive' },
              }}
              size="sm"
              className="disconnect-btn-statusbar"
            />
          )}
          {isDisconnected && hasMessages && (
            <ConnectButton
              onConnect={onConnect}
              onDisconnect={onDisconnect}
              stateContent={{
                disconnected: { children: 'Connect', variant: 'active' },
                ready: { children: 'Disconnect', variant: 'destructive' },
              }}
              size="sm"
              className="connect-btn-statusbar"
            />
          )}
        </div>
      </div>

      <ValidationPanel
        turns={validationTurns}
        collapsed={panelCollapsed}
        onToggle={() => setPanelCollapsed((v) => !v)}
        config={ivrConfig}
        intrinsics={INTRINSICS}
        activeIntrinsic={activeIntrinsic}
        onIntrinsicChange={(checked, id) => {
          const newId = checked ? id : null;
          onIntrinsicChange(newId);
          if (isConnected && client) {
            client.sendClientMessage('set_ivr_validation', { enabled: checked });
            setStatusMessage(checked ? 'IVR Validation enabled.' : 'IVR Validation disabled.');
          }
        }}
      />
      </div>
    </div>
  );
}

function ValidationPanel({
  turns,
  collapsed,
  onToggle,
  config,
  intrinsics,
  activeIntrinsic,
  onIntrinsicChange,
}: {
  turns: ValidationTurn[];
  collapsed: boolean;
  onToggle: () => void;
  config: { requirements: string[]; nSamples: number } | null;
  intrinsics: { id: string; label: string; enabled: boolean }[];
  activeIntrinsic: string | null;
  onIntrinsicChange: (checked: boolean, id: string) => void;
}) {
  const ivrEnabled = activeIntrinsic === 'intrinsic1';
  const active = turns[0];
  const history = turns.slice(1);
  const placeholder = !active && config
    ? makeTurn('__placeholder__', config.nSamples, config.requirements)
    : null;
  const PANEL_WIDTH = 320;
  const TAB_WIDTH = 40;

  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        // Expanded: sit to the right of the chat with a 12px gap.
        // Collapsed: slide behind the chat, leaving only TAB_WIDTH exposed past its right edge.
        left: collapsed
          ? `calc(100% - ${PANEL_WIDTH - TAB_WIDTH}px)`
          : 'calc(100% + 12px)',
        width: `${PANEL_WIDTH}px`,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#141c2e',
        border: '1px solid #263659',
        borderRadius: '8px',
        overflow: 'hidden',
        minHeight: 0,
        // Sit behind the chat panel when collapsed so only the tab pokes out.
        zIndex: collapsed ? 0 : 1,
        transition: 'left 240ms ease',
      }}
    >
      {/* Clickable tab strip on the right edge of the panel.
          When collapsed this is the only part visible past the chat's right edge. */}
      <button
        type="button"
        onClick={onToggle}
        aria-label={collapsed ? 'Expand requirements panel' : 'Collapse requirements panel'}
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          right: 0,
          width: `${TAB_WIDTH}px`,
          border: 'none',
          borderLeft: '1px solid #263659',
          backgroundColor: '#1c2435',
          color: '#99bbd5',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          zIndex: 1,
        }}
      >
        <span
          style={{
            writingMode: 'vertical-rl',
            transform: 'rotate(180deg)',
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.32px',
            fontWeight: 600,
          }}
        >
          Requirements
        </span>
      </button>

      <div
        style={{
          padding: '12px 16px',
          paddingRight: `${TAB_WIDTH + 16}px`,
          borderBottom: '1px solid #263659',
          color: '#ffffff',
          fontSize: '14px',
          fontWeight: 600,
          letterSpacing: '0.16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
        }}
      >
        <span>Requirement validation</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {intrinsics.map(({ id, label, enabled }) => (
            <Toggle
              key={id}
              id={`intrinsic-${id}`}
              aria-label={label}
              labelText=""
              hideLabel
              labelA="Off"
              labelB="On"
              size="sm"
              toggled={activeIntrinsic === id}
              disabled={!enabled}
              onToggle={(checked: boolean) => onIntrinsicChange(checked, id)}
            />
          ))}
        </div>
      </div>

      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px 16px',
          paddingRight: `${TAB_WIDTH + 16}px`,
          minHeight: 0,
          opacity: ivrEnabled ? 1 : 0.4,
          transition: 'opacity 160ms ease',
        }}
      >
        {active && <ValidationGrid turn={active} isActive />}
        {!active && placeholder && <ValidationGrid turn={placeholder} isPlaceholder />}

        {history.length > 0 && (
          <div style={{ marginTop: '16px', borderTop: '1px solid #263659', paddingTop: '12px' }}>
            <div style={{ color: '#99bbd5', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.32px', marginBottom: '8px' }}>
              Previous turns
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {history.map((t) => (
                <ValidationThumbnail key={t.turnId} turn={t} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ValidationGrid({
  turn,
  isActive,
  isPlaceholder,
}: {
  turn: ValidationTurn;
  isActive?: boolean;
  isPlaceholder?: boolean;
}) {
  const { requirements, nSamples, cells, sampleStarted, texts, chosenGen, elapsedMs } = turn;
  const nPassed = cells.filter((row) => row.every((c) => c === 'pass')).length;
  const genPassed = (gen: number) =>
    cells[gen] && cells[gen].length === requirements.length && cells[gen].every((c) => c === 'pass');
  const genFailed = (gen: number) =>
    cells[gen] && cells[gen].some((c) => c === 'fail');

  return (
    <div>
      {/* Column headers */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `1fr repeat(${nSamples}, 36px)`,
          alignItems: 'center',
          columnGap: '8px',
          marginBottom: '6px',
          fontSize: '11px',
          color: '#99bbd5',
          textTransform: 'uppercase',
          letterSpacing: '0.32px',
        }}
      >
        <div />
        {Array.from({ length: nSamples }, (_, gen) => (
          <div
            key={gen}
            style={{
              textAlign: 'center',
              color: chosenGen === gen ? '#32a6ff' : sampleStarted[gen] ? '#ffffff' : '#99bbd5',
              fontWeight: chosenGen === gen ? 700 : 400,
            }}
          >
            Gen {gen + 1}
          </div>
        ))}
      </div>

      {/* Requirement rows */}
      {requirements.map((req, reqIndex) => (
        <div
          key={reqIndex}
          style={{
            display: 'grid',
            gridTemplateColumns: `1fr repeat(${nSamples}, 36px)`,
            alignItems: 'center',
            columnGap: '8px',
            padding: '6px 0',
            borderBottom: '1px solid rgba(38, 54, 89, 0.4)',
            fontSize: '13px',
            color: '#ffffff',
          }}
        >
          <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {req}
          </div>
          {Array.from({ length: nSamples }, (_, gen) => (
            <ValidationCell key={gen} state={cells[gen]?.[reqIndex] ?? 'pending'} />
          ))}
        </div>
      ))}

      {/* Footer: chosen + elapsed */}
      {!isPlaceholder && (
        <div
          style={{
            marginTop: '10px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '12px',
            color: '#99bbd5',
          }}
        >
          <span>
            {elapsedMs == null
              ? (isActive ? 'Validating…' : '')
              : `${nPassed}/${nSamples} passed`}
          </span>
          <span style={{ color: chosenGen != null && chosenGen >= 0 ? '#32a6ff' : '#e0826b' }}>
            {elapsedMs == null
              ? ''
              : chosenGen != null && chosenGen >= 0
                ? `chose Gen ${chosenGen + 1} · ${(elapsedMs / 1000).toFixed(2)}s`
                : `no valid answer · ${(elapsedMs / 1000).toFixed(2)}s`}
          </span>
        </div>
      )}

      {/* Candidate responses */}
      {!isPlaceholder && (
      <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {Array.from({ length: nSamples }, (_, gen) => {
          const text = texts[gen];
          const isChosen = chosenGen === gen;
          const passed = genPassed(gen);
          const failed = genFailed(gen);
          const statusIcon = passed ? '✓' : failed ? '✗' : '•';
          const statusColor = passed ? '#42be65' : failed ? '#fa4d56' : '#4a5e87';
          return (
            <div
              key={gen}
              style={{
                padding: '8px 10px',
                borderRadius: '4px',
                border: isChosen ? '1px solid #32a6ff' : '1px solid rgba(38, 54, 89, 0.6)',
                backgroundColor: isChosen ? 'rgba(50, 166, 255, 0.08)' : 'transparent',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '11px',
                  color: '#99bbd5',
                  textTransform: 'uppercase',
                  letterSpacing: '0.32px',
                  marginBottom: '4px',
                }}
              >
                <span style={{ color: isChosen ? '#32a6ff' : '#99bbd5', fontWeight: isChosen ? 700 : 400 }}>
                  Gen {gen + 1}
                </span>
                <span style={{ color: statusColor, fontWeight: 700 }}>{statusIcon}</span>
                {isChosen && <span style={{ color: '#32a6ff' }}>(chosen)</span>}
              </div>
              <div
                style={{
                  fontSize: '13px',
                  lineHeight: '18px',
                  color: text ? '#ffffff' : '#4a5e87',
                  fontStyle: text ? 'normal' : 'italic',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {text ?? (sampleStarted[gen] ? 'Generating…' : 'Waiting…')}
              </div>
            </div>
          );
        })}
      </div>
      )}
    </div>
  );
}

function ValidationCell({ state }: { state: CellState }) {
  const common: React.CSSProperties = {
    width: '24px',
    height: '24px',
    borderRadius: '4px',
    margin: '0 auto',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '13px',
    fontWeight: 700,
    transition: 'background-color 120ms ease-out, color 120ms ease-out, transform 120ms ease-out',
  };

  if (state === 'pass') {
    return (
      <div style={{ ...common, backgroundColor: 'rgba(66, 190, 101, 0.18)', color: '#42be65' }}>
        ✓
      </div>
    );
  }
  if (state === 'fail') {
    return (
      <div style={{ ...common, backgroundColor: 'rgba(250, 77, 86, 0.18)', color: '#fa4d56' }}>
        ✗
      </div>
    );
  }
  return (
    <div style={{ ...common, backgroundColor: 'rgba(38, 54, 89, 0.5)', color: '#4a5e87' }}>
      •
    </div>
  );
}

function ValidationThumbnail({ turn }: { turn: ValidationTurn }) {
  const { requirements, nSamples, cells, chosenGen, elapsedMs } = turn;
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '11px',
        color: '#99bbd5',
      }}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${nSamples}, 10px)`,
          gridTemplateRows: `repeat(${requirements.length}, 10px)`,
          gap: '2px',
          gridAutoFlow: 'column',
        }}
      >
        {Array.from({ length: nSamples }).flatMap((_, gen) =>
          requirements.map((_, reqIndex) => {
            const s = cells[gen]?.[reqIndex] ?? 'pending';
            const bg =
              s === 'pass' ? 'rgba(66, 190, 101, 0.7)'
              : s === 'fail' ? 'rgba(250, 77, 86, 0.7)'
              : 'rgba(38, 54, 89, 0.8)';
            return (
              <div
                key={`${gen}-${reqIndex}`}
                style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: bg }}
              />
            );
          }),
        )}
      </div>
      <span style={{ whiteSpace: 'nowrap' }}>
        {chosenGen != null && chosenGen >= 0 ? `Gen ${chosenGen + 1}` : '—'}
        {elapsedMs != null ? ` · ${(elapsedMs / 1000).toFixed(2)}s` : ''}
      </span>
    </div>
  );
}
