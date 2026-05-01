import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "AluminatAI — GPU Cost Intelligence";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          background: "linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #0a0a0a 100%)",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "system-ui, sans-serif",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Accent glow */}
        <div
          style={{
            position: "absolute",
            top: "-120px",
            right: "-120px",
            width: "500px",
            height: "500px",
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(34,197,94,0.15) 0%, transparent 70%)",
            display: "flex",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: "-100px",
            left: "-100px",
            width: "400px",
            height: "400px",
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(34,197,94,0.1) 0%, transparent 70%)",
            display: "flex",
          }}
        />

        {/* Logo / brand mark */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            marginBottom: "24px",
          }}
        >
          <div
            style={{
              width: "64px",
              height: "64px",
              borderRadius: "16px",
              background: "linear-gradient(135deg, #22c55e, #16a34a)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "36px",
              fontWeight: 800,
              color: "#fff",
            }}
          >
            A
          </div>
          <span
            style={{
              fontSize: "52px",
              fontWeight: 800,
              color: "#fff",
              letterSpacing: "-1px",
            }}
          >
            AluminatAI
          </span>
        </div>

        {/* Tagline */}
        <div
          style={{
            fontSize: "28px",
            fontWeight: 600,
            color: "#22c55e",
            marginBottom: "16px",
            display: "flex",
          }}
        >
          GPU Cost Intelligence
        </div>

        {/* Description */}
        <div
          style={{
            fontSize: "20px",
            color: "#a3a3a3",
            maxWidth: "700px",
            textAlign: "center",
            lineHeight: 1.5,
            display: "flex",
          }}
        >
          Real-time energy monitoring · Cost attribution · Cloud savings
        </div>

        {/* Bottom bar */}
        <div
          style={{
            position: "absolute",
            bottom: "0",
            left: "0",
            right: "0",
            height: "4px",
            background: "linear-gradient(90deg, transparent, #22c55e, transparent)",
            display: "flex",
          }}
        />

        {/* URL */}
        <div
          style={{
            position: "absolute",
            bottom: "28px",
            fontSize: "16px",
            color: "#525252",
            display: "flex",
          }}
        >
          www.aluminatai.com
        </div>
      </div>
    ),
    { ...size }
  );
}
