"""用解析构造检验审计口径，测试数据不代表实验结果。"""
import unittest
import numpy as np
from marllib.diagnose_pb_slack_coverage import affine, predicted, residuals, state, summarize, physical_value


def row():
    return {'drones':{
        '2':{'pos':[2.,0.,0.],'v_actual':[-1.,0.,0.],'v_requested':[0.,0.],'a_safe':[1.,0.]},
        '3':{'pos':[0.,0.,0.],'v_actual':[1.,0.,0.],'v_requested':[0.,0.],'a_safe':[-1.,0.]}}}


class DiagnosisTest(unittest.TestCase):
    def test_exact_model_has_zero_prediction_error_but_nonzero_state_change(self):
        now=row();p,p2,v,v2=predicted(now);nxt=row()
        for i,pp,vv in [('2',p,v),('3',p2,v2)]:
            nxt['drones'][i]['pos']=pp.tolist();nxt['drones'][i]['v_actual']=vv.tolist()
        legacy,protocol=residuals(now,nxt)
        self.assertGreater(legacy,1e-3);self.assertAlmostEqual(protocol,0.)

    def test_support_equals_box_corner_maximum(self):
        import itertools
        c,b,*_=affine(state(row()))
        maximum=max(c@np.array(x)-b for x in itertools.product([-2.,2.],repeat=4))
        self.assertAlmostEqual(maximum,2*np.abs(c).sum()-b)
        self.assertGreater(maximum,float(c@np.array([1.,0.,-1.,0.])-b))

    def test_nonclosing_omitted_row_has_zero_legacy_slack(self):
        r=row();r['drones']['2']['v_actual']=[1.,0.,0.];r['drones']['3']['v_actual']=[-1.,0.,0.]
        c,b,branch,_,drift=affine(state(r))
        self.assertEqual(branch,'非接近且约束省略');self.assertEqual(float(np.abs(c).sum()-b),0.)
        self.assertGreater(drift,0.)

    def test_nonclosing_inside_clearance_can_be_infeasible(self):
        r=row();r['drones']['2']['pos']=[.1,0.,0.]
        for d in r['drones'].values():d['v_actual']=[0.,0.,0.]
        self.assertEqual(affine(state(r))[2],'非接近但不可行')

    def test_degenerate_state_not_silently_removed(self):
        r=row();r['drones']['2']['pos']=[0.,0.,0.]
        with self.assertRaises(ValueError):affine(state(r))

    def test_omitted_row_encoding_jumps_but_physical_value_is_continuous(self):
        r=row();r['drones']['2']['pos']=[8.,0.,0.]
        r['drones']['3']['v_actual']=[0.,0.,0.]
        r['drones']['2']['v_actual']=[1e-8,0.,0.]
        qplus=state(r)
        r['drones']['2']['v_actual']=[-1e-8,0.,0.]
        qminus=state(r)
        self.assertAlmostEqual(physical_value(qplus,np.zeros(4)),physical_value(qminus,np.zeros(4)),places=6)
        cp,bp,*_=affine(qplus);cm,bm,*_=affine(qminus)
        self.assertGreater(abs(bp-bm),3.)
