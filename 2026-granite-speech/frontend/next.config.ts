import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  transpilePackages: [
    "@pipecat-ai/voice-ui-kit",
    "@pipecat-ai/client-js",
    "@pipecat-ai/client-react",
    "@pipecat-ai/small-webrtc-transport",
  ],
};

export default nextConfig;
