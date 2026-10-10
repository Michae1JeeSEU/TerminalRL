"""Read-only initialization diagnosis; no optimizer or checkpoint loading."""
import json
import torch
from terminal_marl.training import p3_v3_bc_learnability as bc

torch.set_num_threads(1)
root='/home/user/training_outputs/terminal'
data=bc.load_inputs(root)
reference=bc.reference_results(root,data)[91701]['initial']
torch.manual_seed(91701)
model=bc.HierarchicalActorCentralCriticV42(90)
for name,param in model.named_parameters():
    param.requires_grad_(not name.startswith('critic_'))
actual=bc.measure(model,data)
canonical=json.loads(json.dumps(actual,allow_nan=False))
key_types={}
for field in ('target_counts','prediction_counts'):
    original=actual['groups']['overall']['proposal'][field]
    loaded=reference['groups']['overall']['proposal'][field]
    key_types[field]=dict(actual=sorted({type(k).__name__ for k in original}),
        reference=sorted({type(k).__name__ for k in loaded}))
result=dict(status='INITIAL_CHECK_COMPLETE',seed=91701,raw_equal=actual==reference,
    json_form_equal=canonical==reference,key_types=key_types,
    actual_losses={k:actual[k] for k in ('loss','gate_loss','proposal_loss')},
    reference_losses={k:reference[k] for k in ('loss','gate_loss','proposal_loss')},
    optimizer_created=False,parameter_updates=0,environment_resets=0,teacher_queries=0)
print(json.dumps(result,indent=2,allow_nan=False))
