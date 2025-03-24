import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file"""
    # Get the project root directory (2 levels up from this file)
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'
    
    if not env_path.exists():
        raise FileNotFoundError(
            f"No .env file found at {env_path}. "
            "Please copy .env.template to .env and fill in your credentials."
        )
    
    # Load environment variables
    load_dotenv(env_path)
    
    # Return a dictionary of all required environment variables
    return {
        'twitter': {
            'api_key': os.getenv('TWITTER_API_KEY'),
            'api_secret': os.getenv('TWITTER_API_SECRET'),
            'access_token': os.getenv('TWITTER_ACCESS_TOKEN'),
            'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        },
        'alphavantage': {
            'api_key': os.getenv('ALPHAVANTAGE_KEY')
        },
        'edgar': {
            'user_agent': os.getenv('EDGAR_USER_AGENT')
        }
    }

def validate_env(config):
    """Validate that all required environment variables are set"""
    required_vars = {
        'twitter': ['api_key', 'api_secret', 'access_token', 'access_token_secret'],
        'alphavantage': ['api_key'],
        'edgar': ['user_agent']
    }
    
    missing_vars = []
    
    for service, vars in required_vars.items():
        for var in vars:
            if not config[service][var]:
                missing_vars.append(f"{service.upper()}_{var.upper()}")
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    return True 