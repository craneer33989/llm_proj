import math
from types import SimpleNamespace
import pytest

from finance_agent.evaluate_likelihood import aggregate, encode_reference, load_prefix


def test_heldout_likelihood_is_token_weighted_instead_of_averaging_actions():
    result = aggregate([{'target_tokens': 1, 'nll_sum': math.log(2), 'correct_tokens': 1},
                        {'target_tokens': 3, 'nll_sum': 3*math.log(4), 'correct_tokens': 1}])
    assert result['cross_entropy_nats_per_token'] == pytest.approx((math.log(2)+3*math.log(4))/4)
    assert result['perplexity'] == pytest.approx(2**.25*4**.75)
    assert result['teacher_forced_token_accuracy'] == .5


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['add_generation_prompt'] is True
        return 'prefix'

    def __call__(self, text, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return SimpleNamespace(input_ids=[10, 11] if text == 'prefix' else [20, 21, 22])


def test_heldout_reference_includes_eos_and_keeps_prefix_out_of_targets():
    row = {'split': 'test', 'prompt': [], 'completion': [{'role': 'assistant', 'content': 'answer'}]}
    sequence, target = encode_reference(Tokenizer(), row)
    assert sequence == [10, 11, 20, 21, 22, 99]
    assert target == [20, 21, 22, 99]
    row['split'] = 'train'
    with pytest.raises(ValueError, match='held-out'):
        encode_reference(Tokenizer(), row)


def test_likelihood_resume_rejects_changed_tokenization_and_bad_records(tmp_path):
    import json
    path=tmp_path/'saved.json'
    row={'case_id':'test-one','turn':1,'group':'direct','action_type':'search'}
    saved={**row,'target_tokens':2,'correct_tokens':1,'nll_sum':.5}
    path.write_text(json.dumps([saved]),encoding='utf-8')
    assert load_prefix(path,[row],[([1,2,3],[2,3])])==[saved]
    with pytest.raises(ValueError,match='tokenization'):
        load_prefix(path,[row],[([1,2,3,4],[2,3,4])])
    saved['correct_tokens']=3
    path.write_text(json.dumps([saved]),encoding='utf-8')
    with pytest.raises(ValueError,match='Invalid saved'):
        load_prefix(path,[row],[([1,2,3],[2,3])])
