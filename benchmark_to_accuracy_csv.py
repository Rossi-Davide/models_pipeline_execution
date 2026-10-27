"""
Scripts that calculates as percentages accuracy scores 
for the whole dataset, each build, and each subcategory 
"""
import argparse
import pandas as pd

from pathlib import Path
from data.images_store import ImageRecord
from data.query_processor import load_model_embeddings, load_benchmarks_as_df, load_image_records_as_df
from pipeline_modelli.model import Model

def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"'{value}' is not an absolute path")
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write accuracy scores as CSV files")
    parser.add_argument("--output-folder", required=True, type=_absolute_path,
                        help="Absolute path of the folder where the CSV files are written")
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    image_records_df = load_image_records_as_df()

    models = [
        ("clip","openai/clip-vit-base-patch32"),
        ("clip","openai/clip-vit-large-patch14"),
        ("dinov3","facebook/dinov3-vitl16-pretrain-lvd1689m"),
        ("dinov3","facebook/dinov3-vith16plus-pretrain-lvd1689m")
    ]

    for model, checkpoint in models:
        benchmark_scores_df = load_benchmarks_as_df(model, checkpoint)
        df = pd.merge(image_records_df,benchmark_scores_df, 
                    left_on=['folder_path','image_name','modification'], 
                    right_on=['folder','image','mod'],
                    how='right')
        df = df.loc[df['metric'] == 'cosine']
        
        tags = ['reference','-90angle','top','left_side_light','low_light','white_background','wooden_background']
        from benchmark import aggregate_scores
        series = []
        tag_rows = []

        for tag in tags:
            has_tag = df['tags'].fillna('').str.split('|').apply(lambda row_tags: tag in row_tags)
            contains_tag = df.loc[has_tag]
            for kind in ['clean','dirty']:
                has_kind = contains_tag['tags'].fillna('').str.split('|').apply(lambda row_tags: kind in row_tags)
                contains_kind = contains_tag.loc[has_kind]
                series.append(aggregate_scores(contains_kind))
                tag_rows.append(f"{tag}_{kind}")

        out_df = pd.DataFrame(series)
        tag_df = pd.DataFrame(data=tag_rows,columns=['Categories'])
        res = pd.concat([tag_df.reset_index(drop=True),out_df.reset_index(drop=True)],axis=1)
        summary_rows = [('All Dataset', aggregate_scores(df))]
        for build_name in df['build_name'].unique():
            summary_rows.append((build_name, aggregate_scores(df.loc[df['build_name'] == build_name])))
        summary_df = pd.DataFrame([s for _, s in summary_rows])
        summary_df.insert(0, 'Categories', [name for name, _ in summary_rows])
        res = pd.concat([summary_df, res], axis=0, ignore_index=True)

        file_name = f"{model}-{checkpoint}.csv".replace("/", "_")
        path = args.output_folder / file_name
        res.to_csv(path)

if __name__ == "__main__":
    main()