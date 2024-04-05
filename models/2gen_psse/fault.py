# File:"C:\Users\bwang2\OneDrive - NREL\Ongoing\LDRD\LDRD\emt_BW\models\2gen_psse\gentrip.py", generated on TUE, AUG 17 2021  16:59, PSS(R)E release 33.11.00
psspy.read(0,r"""C:\Users\bwang2\Documents\GitHub\emt\models\2gen_psse\2gen_noRE_v33.raw""")
psspy.dyre_new([1,1,1,1],r"""C:\Users\bwang2\Documents\GitHub\emt\models\2gen_psse\2gen_noRE_gast.dyr""","","","")

# psspy.change_plmod_con(1,r"""1""",r"""SEXS""",1, 0.0)
# psspy.change_plmod_con(1,r"""1""",r"""SEXS""",3, 1.0)
# psspy.change_plmod_con(4,r"""1""",r"""SEXS""",3, 1.0)

psspy.cong(0)
psspy.conl(0,1,1,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.conl(0,1,2,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.conl(0,1,3,[0,0],[0.0, 100.0,0.0, 100.0])
psspy.chsb(0, 1, [-1, -1, -1, 1, 7, 0])  # SPEED, machine speed deviation from nominal, pu
psspy.chsb(0, 1, [-1, -1, -1, 1, 2, 0])  # PELEC, machine electric power, pu on SBASE
psspy.chsb(0, 1, [-1, -1, -1, 1, 5, 0])  # EFD, generator main field voltage
psspy.chsb(0, 1, [-1, -1, -1, 1, 6, 0])  # PMECH, turbine mechanical pwoer, pu on MBASE
psspy.chsb(0, 1, [-1, -1, -1, 1, 14, 0])  # voltage and angle

psspy.dynamics_solution_param_2([_i,_i,_i,_i,_i,_i,_i,_i],[_f,_f, 0.00833333,_f,_f,_f,_f,_f])

psspy.strt_2([0,0],r"""C:\Users\bwang2\Documents\GitHub\emt\models\2gen_psse\out.out""")
psspy.run(0, 1,0,1,1)
psspy.dist_bus_fault(1,2, 24.0,[ 10, 0.0])
psspy.run(0, 1.1,0,1,1)
psspy.dist_clear_fault(1)
psspy.run(0, 15.0,0,1,1)
# psspy.dist_bus_fault(1,3, 24.0,[ 0.01, 0.05])
# psspy.dist_bus_fault(1,1, 24.0,[0.0,-0.2E+10])
#psspy.run(0, 0.18333,0,1,1)
#psspy.dist_clear_fault(1)

