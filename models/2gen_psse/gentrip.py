# File:"C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\2gen_psse\gentrip.py", generated on TUE, AUG 17 2021  16:59, PSS(R)E release 33.11.00
psspy.read(0,r"""C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\2gen_psse\2gen_v34.raw""")
psspy.dyre_new([1,1,1,1],r"""C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\2gen_psse\2gen.dyr""","","","")

psspy.cong(0)
psspy.conl(0,1,1,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.conl(0,1,2,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.conl(0,1,3,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.chsb(0,1,[-1,-1,-1,1,1,0])
psspy.chsb(0,1,[-1,-1,-1,1,2,0])
psspy.chsb(0,1,[-1,-1,-1,1,3,0])
psspy.chsb(0,1,[-1,-1,-1,1,4,0])
psspy.chsb(0,1,[-1,-1,-1,1,5,0])
psspy.chsb(0,1,[-1,-1,-1,1,6,0])
psspy.chsb(0,1,[-1,-1,-1,1,7,0])
psspy.chsb(0,1,[-1,-1,-1,1,13,0])
psspy.chsb(0,1,[-1,-1,-1,1,16,0])
psspy.chsb(0, 1, [-1, -1, -1, 1, 11, 0])



psspy.dynamics_solution_param_2([_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f, 0.0083333,_f,_f,_f,_f,_f])
psspy.strt_2([0,0],r"""C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\2gen_psse\out.out""")
psspy.run(0, 0.01,0,1,1)
psspy.dist_machine_trip(4,r"""1""")
# psspy.change_gref(4,r"""1""",0.0)
psspy.run(0, 50,0,1,1)
