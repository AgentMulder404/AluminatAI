import nextConfig from "eslint-config-next";

const config = [
  ...nextConfig,
  {
    ignores: ["agent/**", ".next/**", "node_modules/**", "deploy/**"],
  },
];

export default config;
