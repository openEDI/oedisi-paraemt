import matplotlib.pyplot as plt
import helics as h
import logging
import numpy as np


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def destroy_federate(fed):
    # Adding extra time request to clear out any pending messages to avoid
    #   annoying errors in the broker log. Any message are tacitly disregarded.
    grantedtime = h.helicsFederateRequestTime(fed, h.HELICS_TIME_MAXTIME)
    status = h.helicsFederateDisconnect(fed)
    h.helicsFederateDestroy(fed)
    logger.info("Federate disconnected")


if __name__ == "__main__":
    total_interval = 0.1
    ##########  Registering  federate and configuring from JSON################
    fed = h.helicsCreateValueFederateFromConfig("save_fed_config.json")
    federate_name = h.helicsFederateGetName(fed)
    sub_count = h.helicsFederateGetInputCount(fed)
    pub_count = h.helicsFederateGetPublicationCount(fed)
    # Diagnostics to confirm JSON config correctly added the required
    #   publications and subscriptions
    subid = {}
    for i in range(0, sub_count):
        subid[i] = h.helicsFederateGetInputByIndex(fed, i)
        sub_name = h.helicsSubscriptionGetTarget(subid[i])
        logger.debug(f"Sub {sub_name}")
        
    pubid = {}
    for i in range(0, pub_count):
        pubid[i] = h.helicsFederateGetPublicationByIndex(fed, i)
        pub_name = h.helicsPublicationGetName(pubid[i])

    ##############  Entering Execution Mode  ##################################
    h.helicsFederateEnterExecutingMode(fed)
    logger.info("Entered HELICS execution mode")

    update_interval = int(h.helicsFederateGetTimeProperty(fed, h.HELICS_PROPERTY_TIME_PERIOD))
    grantedtime = 0
    # Data collection lists
    time_sim = []
    total_current = []
    soc = {}
    # As long as granted time is in the time range to be simulated...
    while grantedtime < total_interval:
        # Time request for the next physical interval to be simulated
        requested_time = grantedtime + update_interval
        logger.debug(f"Requesting time {requested_time}")
        grantedtime = h.helicsFederateRequestTime(fed, requested_time)
        logger.debug(f"Granted time {grantedtime}")

        # Iterating over publications in this case since this example
        #  uses only one charging voltage for all five batteries

        for j in range(0, sub_count):
            # Get the applied charging voltage from the EV
            save_term = h.helicsInputGetDouble((subid[j]))
            # save_term = h.helicsInputGetTarget((subid[j]))
            logger.debug(f"save term {save_term:.2f}")
            # logger.debug(f"\tReceived voltage {save_term:.2f}" 
            #             f" from input {h.helicsSubscriptionGetTarget(subid[j])}")
            # Store  for later analysis/graphing
            if subid[j] not in soc:
                soc[subid[j]] = []
            soc[subid[j]].append(float(save_term))
        # Data collection vectors
        time_sim.append(grantedtime)

    # Cleaning up HELICS stuff once we've finished the co-simulation.
    destroy_federate(fed)
    # Printing out final results graphs for comparison/diagnostic purposes.
    xaxis = np.array(time_sim)
    y = []
    for key in soc:
        y.append(np.array(soc[key]))

    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("V and I")

    axs[0].plot(xaxis, y[0], color="tab:blue", linestyle="-")
    # axs[0].set_yticks(np.arange(0, 1.25, 0.5))
    axs[0].set(ylabel="Voltage")
    axs[0].grid(True)

    axs[1].plot(xaxis, y[1], color="tab:blue", linestyle="-")
    axs[1].set(ylabel="Current")
    axs[1].grid(True)

    plt.xlabel("time (hr)")
    plt.savefig("fundamental_default_battery_SOCs.png", format="png")
    plt.show()