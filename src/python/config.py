import os
import json

class Config:
    def __init__(self):
       SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
       CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "config", "path.json")
       config = open(CONFIG_PATH)
       self.config = json.load(config)

    def get_data(self, file):
         return self.config["project_path"]+self.config["data_path"]+"/"+file
    
    def get_model(self, file):
        return self.config["project_path"]+self.config["model_path"]+"/"+file
    
    def get_build(self):
        return self.config["project_path"]+self.config["build_path"]
    