"""Trophic labels for the compartments of the 20 food webs of igraphdata
(Ulanowicz et al.; Pajek food-web collection), from their names only.

Each web is sent to an LLM with its title and the list of its compartment
names (no edges, no flows, no biomass), so that the labels do not depend on
the graph the embeddings read. For every compartment it returns the expanded
name, the trophic guild (the role) and a coarse taxonomic class, with a
confidence. Input, Output and Respiration are boundary nodes and are left out.

    python foodweb_labels.py            # writes data/foodwebs/labels.csv
"""
import os
from typing import List, Literal

import pandas as pd
from pydantic import BaseModel

from aiquery import ask_dataframe
from aiquery.core.classes import QueryConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "foodwebs")
MODEL = "openai-gpt-6-astra@low"

Guild = Literal["primary_producer", "nonliving_detritus", "microbial",
                "herbivore", "detritivore", "omnivore",
                "invertivore", "top_predator"]
Taxon = Literal["plant_or_alga", "bacteria_or_protist", "invertebrate", "fish",
                "amphibian", "reptile", "bird", "mammal", "nonliving"]


class Compartment(BaseModel):
    index: int
    expanded_name: str
    guild: Guild
    taxon: Taxon
    confidence: Literal["high", "medium", "low"]


class WebLabels(BaseModel):
    compartments: List[Compartment]


PROMPT = """You are an aquatic and wetland ecologist. Below is the list of
compartments of a published carbon-flow food web: "{title}".
Names are sometimes abbreviated: expand them using the ecosystem and the
other names of the list.

For EVERY compartment (use its index), give:
- expanded_name: the full name;
- guild, the trophic role of the compartment as a consumer in this ecosystem,
  from general ecological knowledge of the taxa (not from any data):
  primary_producer (phytoplankton, algae, plants, plant parts such as roots,
  leaves or wood), nonliving_detritus (particulate or dissolved organic
  carbon, sediment carbon, detritus pools), microbial (bacteria and
  heterotrophic protists: flagellates, ciliates), herbivore (mainly eats
  living producers, including grazing zooplankton), detritivore (mainly eats
  detritus or sediment, including deposit and suspension feeders that live on
  detritus), omnivore (substantial plant and animal food), invertivore
  (carnivore mainly of invertebrates or zooplankton, including planktivorous
  and benthivorous fish and insectivorous birds), top_predator (carnivore
  that mainly eats vertebrates: piscivorous fish, raptors, alligators,
  sharks, dolphins, big cats...);
- taxon: plant_or_alga, bacteria_or_protist, invertebrate, fish, amphibian,
  reptile, bird, mammal or nonliving;
- confidence: high, medium or low.
Mixed or aggregated compartments get the guild of most of their carbon
intake. Skip the boundary compartments named Input, Output and Respiration.

Compartments (index: name):
{listing}"""


def main():
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))
    webs = pd.read_csv(os.path.join(DATA, "webs.csv"))
    inner = nodes[nodes.eco <= 2]
    webs["listing"] = [
        "\n".join(f"{r.node}: {r.name}" for r in inner[inner.web == w].itertuples())
        for w in webs.web]
    out = ask_dataframe(webs, PROMPT, model=MODEL, response_format=WebLabels,
                        columns=["web"], max_workers=20, desc="Labelling webs",
                        query_config=QueryConfig(timeout=3600))
    rows = []
    for web, answer in zip(out.web, out.response):
        if answer is None:
            print(f"{web}: no answer", flush=True)
            continue
        for c in answer["compartments"]:
            rows.append({"web": web, **c})
    labels = pd.DataFrame(rows).rename(columns={"index": "node"})
    labels = inner[["web", "node", "name"]].merge(labels, on=["web", "node"], how="left")
    labels.to_csv(os.path.join(DATA, "labels.csv"), index=False)
    print(f"{labels.guild.notna().sum()} / {len(labels)} labelled; cost "
          f"{out.response_cost.sum():.2f}", flush=True)
    print(labels.guild.value_counts().to_string())
    print(labels.confidence.value_counts().to_string())


if __name__ == "__main__":
    main()
