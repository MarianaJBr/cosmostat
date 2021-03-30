from cosmostat.cosmology import get_dataset
from cosmostat.models.one import Params, model

params = Params(0, 0.3, 0.01)


# TODO: Improve tests.

def test_wz():
    """"""
    wz = model.wz
    wz_v = wz(0, params)
    assert wz_v == -1


def test_r_bao():
    """"""
    dataset = get_dataset("BAO")
    chi2_bao = model.make_likelihood(dataset).chi_square_bao
    chi2_bao_v = chi2_bao(params)
    print(chi2_bao_v)
    assert chi2_bao_v == 0
