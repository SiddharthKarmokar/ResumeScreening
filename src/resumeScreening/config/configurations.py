from src.resumeScreening.constants import *
from src.resumeScreening.utils.common import read_yaml, create_directories
from src.resumeScreening.entity import *

class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH,
                 params_path=PARAMS_FILE_PATH):
        self.config=read_yaml(config_path)
        self.params=read_yaml(params_path)

        create_directories([self.config.artifacts_root])
    
    def get_model_pipeline_config(self)->ModelPipelineConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        model_pipeline_config = ModelPipelineConfig(
            root_dir=config.root_dir,
            career_instructions_path: artifacts/career_instructions/
            resume_instructions_path: artifacts/resume_instructions/
        )
        return model_pipeline_config