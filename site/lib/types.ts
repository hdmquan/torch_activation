export type Tag = "smooth" | "monotonic" | "bounded" | "overflow-safe";

export type Param = {
  name: string;
  default: number | boolean;
  type: "float" | "int" | "bool";
};

export type Activation = {
  name: string;
  family: string;
  module: string;
  description: string;
  formula: string;
  params: Param[];
  tags: Tag[];
  paper_ref: string;
  plot: string;
  similar: string[];
};

export type Family = {
  id: string;
  label: string;
  headline: string[];
};

export type SiteData = {
  activations: Activation[];
  families: Family[];
};
