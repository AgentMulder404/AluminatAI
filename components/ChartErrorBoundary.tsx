"use client";

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallbackHeight?: string;
}

interface State {
  hasError: boolean;
}

export default class ChartErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error("Chart render error:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          className={`flex items-center justify-center bg-neutral-900/50 rounded-lg border border-neutral-800 ${
            this.props.fallbackHeight ?? "h-[200px]"
          }`}
        >
          <div className="text-center">
            <p className="text-sm text-neutral-500">Chart failed to load</p>
            <button
              onClick={() => this.setState({ hasError: false })}
              className="text-xs text-indigo-400 hover:text-indigo-300 mt-2"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
