# File:"C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\2gen_psse\gentrip.py", generated on TUE, AUG 17 2021  16:59, PSS(R)E release 33.11.00
psspy.read(0,r"""C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\9bus_psse\ieee9.raw""")
psspy.fnsl([0,0,0,1,1,0,99,0])
psspy.dyre_new([1,1,1,1],r"""C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\9bus_psse\ieee9.dyr""","","","")

psspy.cong(0)
psspy.conl(0,1,1,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.conl(0,1,2,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.conl(0,1,3,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.chsb(0, 1, [-1, -1, -1, 1, 14, 0]) # voltage and angle
psspy.chsb(0, 1, [-1, -1, -1, 1, 2, 0])  # PELEC, machine electric power, pu on SBASE
psspy.chsb(0, 1, [-1, -1, -1, 1, 3, 0])  # QELEC, machine reactive power
psspy.chsb(0, 1, [-1, -1, -1, 1, 7, 0])  # Speed
psspy.chsb(0, 1, [-1, -1, -1, 1, 21, 0])  # ITERM

psspy.dynamics_solution_param_2([_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f, 0.001,_f,_f,_f,_f,_f])
psspy.strt_2([0,0],r"""C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\9bus_PSSE\out.out""")
psspy.run(0, 0.1,0,1,1)
psspy.dist_machine_trip(3,r"""1""")
# psspy.change_gref(4,r"""1""",0.0)
psspy.run(0, 5.1,0,1,1)
