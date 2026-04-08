import os
import warnings


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["TF_USE_LEGACY_KERAS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

from educational_ai_analytics import config  
