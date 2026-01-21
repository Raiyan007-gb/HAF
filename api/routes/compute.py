"""Computation endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas.requests import ComputeRequest
from api.schemas.responses import ComputeResponse
from core.models.haf_config import HAFConfig
from services.haf_service import HAFService
from utils import helpers as hp
from pathlib import Path
import pickle

router = APIRouter()


def save_sample_results(results, sample_ix, model_name, data_name, explicit_prompting):
    """Save sample results to file"""
    from utils.data_path_prefixes import HAF_RESULTS_PATH
    
    if explicit_prompting == '':
        directory_path = Path(HAF_RESULTS_PATH + "_naive" + "/" + model_name.split('/')[1] + '/' + data_name + '/')
    else:
        directory_path = Path(HAF_RESULTS_PATH + "/" + model_name.split('/')[1] + '/' + data_name + '/')
    
    directory_path.mkdir(parents=True, exist_ok=True)
    file_path = directory_path / (str(sample_ix) + '.pkl')
    
    with file_path.open("wb") as f:
        pickle.dump(results, f)


def compute_haf_for_model_dataset(model_name: str, data_name: str, config: HAFConfig):
    """Compute HAF metrics for a specific model and dataset"""
    service = HAFService(config)
    
    # Load output tokens and parsed outputs
    output_tokens_dict = hp.get_output_tokens(model_name, data_name, config.explicit_prompting)
    parsed_output_dict = hp.get_parsed_outputs(model_name, data_name, config.explicit_prompting)
    
    samples_computed = 0
    
    # Process each sample
    for sample_ix in range(len(parsed_output_dict['initial']['input_texts'])):
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
        samples_computed += 1
    
    return samples_computed


@router.post("/single", response_model=ComputeResponse)
async def compute_single(request: ComputeRequest, background_tasks: BackgroundTasks):
    """
    Compute HAF metrics for a single model/dataset combination
    This runs in the background
    """
    try:
        # Create configuration
        config = HAFConfig(
            explicit_prompting='_explicit' if request.config.explicit_prompting else '',
            use_scores=request.config.use_scores,
            similarity_model=request.config.similarity_model
        )
        
        # Add computation to background tasks
        background_tasks.add_task(
            compute_haf_for_model_dataset,
            request.model_name,
            request.data_name,
            config
        )
        
        return ComputeResponse(
            success=True,
            message=f"Computation started for {request.model_name} on {request.data_name}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.post("/all", response_model=ComputeResponse)
async def compute_all(background_tasks: BackgroundTasks, 
                     explicit_prompting: bool = True,
                     use_scores: bool = False,
                     similarity_model: str = "cross-encoder/stsb-distilroberta-base"):
    """
    Compute HAF metrics for all model/dataset combinations
    This runs in the background
    """
    try:
        import json
        
        # Load model and data details
        with open("utils/model_size_map.json", "r") as file:
            model_size = json.load(file)
        with open("utils/data_path_map.json", "r") as file:
            data_path = json.load(file)
        
        data_names = list(data_path.keys())
        model_names = list(model_size.keys())
        
        # Create configuration
        config = HAFConfig(
            explicit_prompting='_explicit' if explicit_prompting else '',
            use_scores=use_scores,
            similarity_model=similarity_model
        )
        
        # Add computation tasks for all combinations
        for data_name in data_names:
            for model_name in model_names:
                background_tasks.add_task(
                    compute_haf_for_model_dataset,
                    model_name,
                    data_name,
                    config
                )
        
        total_combinations = len(data_names) * len(model_names)
        
        return ComputeResponse(
            success=True,
            message=f"Computation started for {total_combinations} model/dataset combinations"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")
