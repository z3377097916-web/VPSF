import contextlib,copy,json,math,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'trainer')]
import numpy as np
import torch
from preference.selectors import linear,deduplicate,choose
from llava_trainer import LlavaDPOTrainer
from base_trainer import BaseDPOTrainer
from data import LazySupervisedDataset,DataCollatorForSupervisedDataset

class ContractTests(unittest.TestCase):
    def test_weighted_image_objective(self):
        rs=[dict(image_id=i,prompt='x',chosen=c,rejected='z',roles=['A']) for i,c in [('a','c'),('a','d'),('b','e')]]
        rs=deduplicate(rs);self.assertEqual([r['pair_weight'] for r in rs],[.75,.75,1.5])
        loss=torch.tensor([2.,4.,9.]);w=torch.tensor([r['pair_weight'] for r in rs]);self.assertEqual((loss*w).mean().item(),6.)
    def test_role_dedup_is_within_image(self):
        rs=[dict(image_id=i,prompt='x',chosen='c',rejected='r',roles=[role]) for i,role in [('a','A'),('a','B'),('b','A')]]
        result=deduplicate(rs);self.assertEqual(len(result),2);self.assertEqual(result[0]['roles'],['A','B'])
    def test_linear_reject_choice_is_not_chosen_conditioned(self):
        rng=np.random.default_rng(1);mu=rng.normal(size=50);sd=rng.uniform(.1,2,size=50);w=rng.normal(size=51)
        model=dict(coef=w.tolist(),mu=mu.tolist(),sd=sd.tolist());rr=rng.normal(size=(22,25));chosen=rng.normal(size=(2,25))
        ranks=[]
        for c in chosen:ranks.append(np.argsort([linear(model,np.r_[r,r-c]) for r in rr]))
        np.testing.assert_array_equal(*ranks)
    def test_logps_mask_and_causal_shift(self):
        t=BaseDPOTrainer.__new__(BaseDPOTrainer);t.is_encoder_decoder=False;t.label_pad_token_id=-100
        logits=torch.randn(2,7,13,requires_grad=True);labels=torch.tensor([[-100,-100,-100,2,3,4,-100],[-100,-100,5,6,7,8,9]])
        got=t._get_batch_logps(logits,labels);target=[]
        for b in range(2):target.append(sum(logits[b,j-1].log_softmax(-1)[labels[b,j]] for j in range(1,7) if labels[b,j]!=-100))
        torch.testing.assert_close(got,torch.stack(target));got.sum().backward();self.assertEqual(logits.grad[0,0].abs().sum().item(),0.)
    def test_dpo_gradient_direction(self):
        t=BaseDPOTrainer.__new__(BaseDPOTrainer);t.accelerator=SimpleNamespace(device=torch.device('cpu'));t.dpo_beta=.1
        c=torch.tensor([0.],requires_grad=True);r=torch.tensor([0.],requires_grad=True)
        losses,_,_=t.dpo_loss(c,r,torch.zeros(1),torch.zeros(1));self.assertAlmostEqual(losses.item(),math.log(2),6)
        losses.backward();self.assertLess(c.grad.item(),0);self.assertGreater(r.grad.item(),0)
    def test_weights_actually_used_by_trainer(self):
        t=LlavaDPOTrainer.__new__(LlavaDPOTrainer);t.compute_ref_log_probs=lambda x:(torch.zeros(2),torch.zeros(2))
        t.dpo_concatenated_forward=lambda model,inputs:(torch.zeros(2),torch.zeros(2),torch.tensor(0.),torch.tensor(0.))
        t.dpo_loss=lambda *args:(torch.tensor([2.,6.]),torch.zeros(2),torch.zeros(2))
        loss,_=t.get_batch_loss_metrics(None,{'pair_weight':torch.tensor([.5,1.5])});self.assertEqual(loss.item(),5.)
    def test_reference_disables_and_restores_adapter(self):
        class Fake:
            active=True
            @contextlib.contextmanager
            def disable_adapter(self):
                self.active=False
                try:yield
                finally:self.active=True
        t=LlavaDPOTrainer.__new__(LlavaDPOTrainer);t.model=Fake();t.ref_model=None;t.is_peft_model=True;t.ref_adapter_name=None
        t.accelerator=SimpleNamespace(unwrap_model=lambda x:x)
        def forward(m,x):
            self.assertFalse(m.active);self.assertFalse(torch.is_grad_enabled());return (torch.tensor(1),torch.tensor(2),None,None)
        t.dpo_concatenated_forward=forward;t.compute_ref_log_probs({});self.assertTrue(t.model.active)
    def test_collator_attention_keeps_unk_in_answer(self):
        tok=SimpleNamespace(pad_token_id=0);r=dict(chosen_input_ids=torch.tensor([1,0,2]),chosen_labels=torch.tensor([-100,0,2]),reject_input_ids=torch.tensor([1,3]),reject_labels=torch.tensor([-100,3]),images=torch.zeros(3,2,2),pair_weight=1)
        b=DataCollatorForSupervisedDataset(tok)([r]);self.assertTrue(b['chosen_attention_mask'].all())

if __name__=='__main__':unittest.main()
