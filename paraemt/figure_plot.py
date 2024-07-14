import matplotlib.pyplot as plt
import numpy as np

def results_plot(self, ts,Tlen):
    # Plot the results ==========================================================================
    xaxis = np.arange(0, Tlen, ts)
    fig, axs = plt.subplots(4,3, sharex=True, sharey=False)

    # No 1, synchronous generator Delta angle
    fig.suptitle("Generator Rotor Angle")
    axs[0,0].plot(xaxis, self.emt.x, color="tab:red", linestyle="-")
    axs[0,0].set_yticks(np.arange(-1.5, 1.5, 0.5))
    # axs[0,0].set_ylim(-1.5, 1.5)
    axs[0,0].set(ylabel="Generator Rotor Angle (rad)")
    axs[0,0].grid(True)


    # No 2, synchronous generator rotor frequency


    # No m, one bus voltage
    # fig.suptitle("V and I")
    # bus_num=int(voltage_term.shape[1]/3)
    # print(bus_num)
    # bus_idx=1  # Choose bus index
    # axs[0,0].plot(xaxis, voltage_term[:,bus_idx], color="tab:red", linestyle="-")
    # axs[0,0].plot(xaxis, voltage_term[:,bus_idx+bus_num], color="tab:green", linestyle="-")
    # axs[0,0].plot(xaxis, voltage_term[:,bus_idx+2*bus_num], color="tab:blue", linestyle="-")
    # axs[0,0].set_yticks(np.arange(-1.5, 1.5, 0.5))
    # axs[0,0].set_xlim(0, time_sim[-1])
    # axs[0,0].set_ylim(-1.5, 1.5)
    # axs[0,0].set(ylabel="Voltage")
    # axs[0,0].grid(True)

    # # No n, one branch current
    # bran_num=int(current_term.shape[1]/3)
    # print(bran_num)
    # bran_idx=2  # Choose branch index
    # axs[0,1].plot(xaxis, current_term[:,0+9*(bran_idx-1)], color="tab:red", linestyle="-")
    # axs[0,1].plot(xaxis, current_term[:,1+9*(bran_idx-1)], color="tab:green", linestyle="-")
    # axs[0,1].plot(xaxis, current_term[:,2+9*(bran_idx-1)], color="tab:blue", linestyle="-")
    # # axs[1].set_yticks(np.arange(-8, 8, 2))
    # axs[0,1].set_xticks(np.arange(0, 0.2, 0.02))
    # axs[0].set_xlim(0, time_sim[-1])
    # axs[1].set_ylim(-15, 15)
    # axs[1].set(ylabel="Current")
    # axs[1].grid(True)

    plt.xlabel("time (s)")
    plt.savefig("ParaEMT_Voltage_Current.png", format="png")
    plt.show()
