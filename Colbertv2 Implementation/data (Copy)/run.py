from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            bsize=32,
            root="/experiments",
        )
        trainer = Trainer(
            triples="data/query_triplets.jsonl",
            queries="data/queries.tsv",
            collection="data/corpus.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")