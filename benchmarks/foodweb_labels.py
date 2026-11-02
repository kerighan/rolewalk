"""Trophic labels for the compartments of the 20 food webs of igraphdata
(Ulanowicz et al.; Pajek food-web collection), from their names only.

Each web is sent to an LLM with its title and the list of its compartment
names (no edges, no flows, no biomass), so that the labels do not depend on
the graph the embeddings read. For every compartment it returns the expanded
name, the trophic guild (the role) and a coarse taxonomic class, with a
confidence. Input, Output and Respiration are boundary nodes and are left out.

    python foodweb_labels.py            # writes data/foodwebs/labels.csv
    python foodweb_labels.py --model anthropic-claude-opus-5-5@high --output labels_opus.csv
"""
import argparse
import os
from typing import List, Literal

import pandas as pd
from pydantic import BaseModel

from concurrent.futures import ThreadPoolExecutor

from aiquery import ask_model
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", default="labels.csv")
    args = parser.parse_args()
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"))
    webs = pd.read_csv(os.path.join(DATA, "webs.csv"))
    inner = nodes[nodes.eco <= 2]
    webs["listing"] = [
        "\n".join(f"{r.node}: {r.name}" for r in inner[inner.web == w].itertuples())
        for w in webs.web]
    path = os.path.join(DATA, args.output)
    done = pd.read_csv(path) if os.path.exists(path) else None
    todo = [w for w in webs.web
            if done is None or done[done.web == w].guild.isna().all()]
    print(f"{len(todo)} webs to label", flush=True)

    def ask(web):
        row = webs[webs.web == web].iloc[0]
        try:
            answer = ask_model(PROMPT.format(title=row.title, listing=row.listing),
                               model=args.model, response_format=WebLabels,
                               query_config=QueryConfig(timeout=3600))
        except Exception as e:  # reported below as no answer
            print(f"{web}: {e}", flush=True)
            return web, None
        print(f"{web} done", flush=True)
        return web, answer.content.model_dump() if answer and answer.content else None

    with ThreadPoolExecutor(20) as pool:
        answers = list(pool.map(ask, todo))
    rows = [] if done is None else done[~done.web.isin(todo)].dropna(
        subset=["guild"])[["web", "node", "expanded_name", "guild", "taxon",
                           "confidence"]].to_dict("records")
    for web, answer in answers:
        if answer is None:
            print(f"{web}: no answer", flush=True)
            continue
        for c in answer["compartments"]:
            rows.append({"web": web, **c})
    rows = [{("node" if k == "index" else k): v for k, v in r.items()} for r in rows]
    labels = pd.DataFrame(rows)
    labels = inner[["web", "node", "name"]].merge(labels, on=["web", "node"], how="left")
    labels.to_csv(path, index=False)
    print(f"{labels.guild.notna().sum()} / {len(labels)} labelled", flush=True)
    print(labels.guild.value_counts().to_string())
    print(labels.confidence.value_counts().to_string())


if __name__ == "__main__":
    main()
