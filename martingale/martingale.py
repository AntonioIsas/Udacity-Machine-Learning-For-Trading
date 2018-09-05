"""Assess a betting strategy.  		   	  			    		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			    		  		  		    	 		 		   		 		  

Template code for CS 4646/7646  		   	  			    		  		  		    	 		 		   		 		  

Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.  		   	  			    		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			    		  		  		    	 		 		   		 		  

-----do not edit anything above this line---  		   	  			    		  		  		    	 		 		   		 		  

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""  		   	  			    		  		  		    	 		 		   		 		  

import numpy as np
import matplotlib.pyplot as plt

def author():
        return 'tb34' # replace tb34 with your Georgia Tech username.  		   	  			    		  		  		    	 		 		   		 		  


def gtid():
    return 900897987 # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
        return result


def run_bets(spins, limit=256):
    win_prob = 0.4736 # set appropriately to the probability of a win

    wins = 0
    loss = 0
    episode_winnings = 0
    bet_amount = 1
    winnings = np.zeros(spins)
    for i in range(spins):
        if episode_winnings >= 80:
            winnings[i]=80
            continue
        if episode_winnings <= -limit:
            winnings[i]=-limit
            continue

        if bet_amount > limit-episode_winnings:
            bet_amount = limit-episode_winnings

        spin = get_spin_result(win_prob)
        if spin:
            wins += 1
            episode_winnings += bet_amount
            winnings[i] = episode_winnings
            bet_amount = 1
        else:
            loss += 1
            episode_winnings -= bet_amount
            winnings[i] = episode_winnings
            bet_amount *= 2

    print(f"Spins {spins} wins={wins}({(wins/spins)*100:.2f}%) loss={loss}({(loss/spins)*100:.2f}%) winnings={episode_winnings}")
    return winnings


def test_code():
    #Figure 1
    # for i in range(3):
    #     run = run_bets(10)
    #     print(run)
    #     plt.plot(run)
    #
    # plt.xlim((0, 300))
    # plt.ylim((-256, 100))
    # plt.show()

    # Figure 2
    bets = 1000
    results = np.empty((0, bets))

    plt.figure(0)
    for i in range(1000):
        run = run_bets(bets)
        plt.plot(run)

        results = np.append(results, [run], axis=0)

    print(results)
    print("------")

    plt.xlim((0, 300))
    plt.ylim((-300, 100))
    plt.show()

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    upper_band = mean + std * 2
    lower_band = mean - std * 2


    plt.figure(1)
    plt.plot(mean)
    plt.plot(upper_band)
    plt.plot(lower_band)
    plt.xlim((0, 300))
    plt.ylim((-300, 100))
    plt.show()

    # Figure 3
    median = np.median(results, axis=0)

    upper_band = median + std*2
    lower_band = median - std*2

    plt.figure(1)
    plt.plot(median)
    plt.plot(upper_band)
    plt.plot(lower_band)
    plt.xlim((0, 300))
    plt.ylim((-300, 100))
    plt.show()

if __name__ == "__main__":
    test_code()
