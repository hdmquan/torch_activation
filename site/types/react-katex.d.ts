declare module "react-katex" {
  import { ReactNode } from "react";

  export interface MathComponentProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
  }

  export const InlineMath: (props: MathComponentProps) => JSX.Element;
  export const BlockMath: (props: MathComponentProps) => JSX.Element;
}
