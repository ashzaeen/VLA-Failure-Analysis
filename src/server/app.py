
from fastapi import FastAPI, HTTPException


from src.server.configs import ServerConfig
from src.server.registry import PolicyRegistry
from src.utils.logger import get_logger
logger = get_logger(__name__)


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title='objectVLA Policy Server')
    registry = PolicyRegistry()
    registry.register_from_config(config)
    app.state.registry = registry

    @app.get('/health')
    def health():
        return {'status': 'ok', 'policies': registry.list_policy_ids()}

    @app.post('/predict')
    def predict(payload: dict):
        try:
            adapter = registry.get(payload['policy_id'])
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        logger.info(f"Received predict request for policy_id: {payload['policy_id']}")  
        result = adapter.predict(canonical_observation = payload['observation'], 
                                 include_attentions=payload.get('include_attentions', False),
                                  batch_size=payload.get('batch_size', 1),
                                  include_nll=payload.get('include_nll', False),
                                  nll_hutchinson_samples=payload.get('nll_hutchinson_samples', 1),
                                  include_obs_embs=payload.get('include_obs_embs', False),
                                  )
        logger.info(f"Predict result: {result.keys()}")  
        actions = result['actions']
        if not isinstance(actions, list):
            raise HTTPException(status_code=500, detail='Adapter returned invalid actions payload.')
        return {
            'actions': actions,
            'raw_actions': result.get('raw_actions'),
            'did_replan': result.get('did_replan', False),
            'attention_last_layer_mean': result.get('attention_last_layer_mean'),
            'v_ts': result.get('v_ts'),
            "nll_res": result.get('nll_res'),
            "obs_embs": result.get('obs_embs'),
        }

    return app