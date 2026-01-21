"""Modular HAF computation script"""
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

from core.models.haf_config import HAFConfig
from services.haf_service import HAFService
from utils import helpers as hp
from utils.data_path_prefixes import HAF_RESULTS_PATH


def save_sample_results(results, sample_ix, model_name, data_name, explicit_prompting):
    """Save sample results to file"""
    if explicit_prompting == '':
        directory_path = Path(HAF_RESULTS_PATH + "_naive" + "/" + model_name.split('/')[1] + '/' + data_name + '/')
    else:
        directory_path = Path(HAF_RESULTS_PATH + "/" + model_name.split('/')[1] + '/' + data_name + '/')
    
    directory_path.mkdir(parents=True, exist_ok=True)
    file_path = directory_path / (str(sample_ix) + '.pkl')
    
    with file_path.open("wb") as f:
        pickle.dump(results, f)


def compute_haf_metrics(config: HAFConfig):
    """
    Compute HAF metrics for all models and datasets
    
    Args:
        config: HAF configuration
    """
    service = HAFService(config)
    
    for data_name in service.data_names:
        for model_name in service.model_names:
            print(f"Processing {model_name} on {data_name} data")
            service.logger.info(f"Processing {model_name} on {data_name} data")
            
            # Load output tokens and parsed outputs
            output_tokens_dict = hp.get_output_tokens(model_name, data_name, config.explicit_prompting)
            parsed_output_dict = hp.get_parsed_outputs(model_name, data_name, config.explicit_prompting)
            
            # Process each sample
            for sample_ix in tqdm(range(len(parsed_output_dict['initial']['input_texts']))):
                # Compute all metrics for this sample
                sample_result = service.compute_sample(
                    sample_ix, model_name, data_name,
                    output_tokens_dict, parsed_output_dict
                )
                
                # Save results
                save_sample_results(
                    sample_result, sample_ix, model_name, data_name,
                    config.explicit_prompting
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute HAF metrics using modular architecture")
    parser.add_argument(
        "--explicit_prompting", type=str, required=False, default='True',
        help="Prompt with explicit instructions (True/False)"
    )
    parser.add_argument(
        "--use_scores", type=str, required=False, default='False',
        help="Use entropy of scores instead of logits (True/False)"
    )
    parser.add_argument(
        "--similarity_model", type=str, required=False,
        default='cross-encoder/stsb-distilroberta-base',
        help="Semantic similarity model name"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = HAFConfig.from_args(
        explicit_prompting=(args.explicit_prompting == 'True'),
        use_scores=(args.use_scores == 'True'),
        similarity_model=args.similarity_model
    )
    
    # Compute metrics
    compute_haf_metrics(config)
