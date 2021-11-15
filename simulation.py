import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 28})

np.random.seed(seed=300)

def valid_assignment(As):
    A1 = As[0]
    A2 = As[1]
    A3 = As[2]
    for i in range(len(A1)):
        if len({A1[i], A2[i], A3[i]}) < 3:
            return False

def generate_assignment(num_paper):
    A1 = np.random.permutation(num_paper)
    A2 = np.random.permutation(num_paper)
    A3 = np.random.permutation(num_paper)
    return (A1, A2, A3)

def find_assignment(num_paper):
    As = generate_assignment(num_paper)
    while valid_assignment(As) == False:
        As = generate_assignment(num_paper)
    
    return As


def gaussian_variables(num_paper, noise_variance):
    ks = np.random.exponential(scale=1.0, size=num_paper)
    #use the line below for reviewers all with the same scalar = 1 
    #ks = np.ones(num_paper)
   
    #b of each reviewer
    bs = np.random.normal(loc=0.0, scale=0.5, size=num_paper)
    
    #qualities of papers
    qualities = np.random.normal(loc=0.0, scale=1.0, size=num_paper)
   
    #[[noise of first paper reviewed by reviewer 0,  noise of first paper reviewed by reviewer 1,  ...],
    # [noise of second paper reviewed by reviewer 0, noise of second paper reviewed by reviewer 1, ...],
    # [noise of third paper reviewed by reviewer 0,  noise of third paper reviewed by reviewer 1,  ...]]
    noises = np.random.normal(loc=0.0, scale=noise_variance, size=[3,num_paper])
    
    true_rank = [sorted(qualities).index(x) for x in qualities]
    return (ks, bs, qualities, noises, true_rank)

def method_1(num_paper, As, gaussians):
    (ks, bs, qualities, noises, true_rank) = gaussians
    (A1, A2, A3) = As
    scores = []
    for i in range(num_paper):
        r1 = np.where(A1==i)[0][0]
        r2 = np.where(A2==i)[0][0]
        r3 = np.where(A3==i)[0][0]
        total_overhead = sum(bs[[r1,r2,r3]]) + noises[0][r1] + noises[1][r2] + noises[2][r3]
        scores.append(((sum(ks[[r1,r2,r3]]))*qualities[i]+total_overhead)/3)
    rank = [sorted(scores).index(x) for x in scores]
    return rank


def method_2(num_paper, As, gaussians):
    (ks, bs, qualities, noises, true_rank) = gaussians
    (A1, A2, A3) = As
    mean_score = []
    stds = []
    for i in range(num_paper): 
        mean_score_i = bs[i] + ks[i]*(qualities[A1[i]]+qualities[A2[i]]+qualities[A3[i]])/3 + sum(noises[:,i])/3
        mean_score.append(mean_score_i)
        stds.append(np.std([ks[i]*qualities[A1[i]]+noises[0][i], ks[i]*qualities[A2[i]]+noises[1][i], ks[i]*qualities[A3[i]]+noises[2][i]]))

    scores = []
    for i in range(num_paper):
        r1 = np.where(A1==i)[0][0]
        r2 = np.where(A2==i)[0][0]
        r3 = np.where(A3==i)[0][0]
        total_scores = [(ks[r1]*qualities[i]+bs[r1]+noises[0][r1]-mean_score[r1])/stds[r1],
                        (ks[r2]*qualities[i]+bs[r2]+noises[1][r2]-mean_score[r2])/stds[r2],
                        (ks[r3]*qualities[i]+bs[r3]+noises[2][r3]-mean_score[r3])/stds[r3]]
        scores.append(sum(total_scores)/3)
    rank = [sorted(scores).index(x) for x in scores]
    return rank


def method_3(num_paper, As, gaussians):
    (ks, bs, qualities, noises, true_rank) = gaussians
    (A1, A2, A3) = As
    scores = []
    for i in range(num_paper):
        r1 = np.where(A1==i)[0][0]
        r2 = np.where(A2==i)[0][0]
        r3 = np.where(A3==i)[0][0]
        numerator = ks[r1]*(ks[r1]*qualities[i]+noises[0][r1]) + ks[r2]*(ks[r2]*qualities[i]+noises[1][r2]) + ks[r3]*(ks[r3]*qualities[i]+noises[2][r3])
        denominator = ks[r1]**2 + ks[r2]**2 + ks[r3]**2
        scores.append(numerator/denominator)
    rank = [sorted(scores).index(x) for x in scores]
    return rank



def kt_distance(num_paper, noise_variance):
    As = find_assignment(num_paper)
    gaussians = gaussian_variables(num_paper, noise_variance)
    (ks, bs, qualities, noises, true_rank) = gaussians
    rank1 = method_1(num_paper, As, gaussians)
    rank2 = method_2(num_paper, As, gaussians)
    rank3 = method_3(num_paper, As, gaussians)
    kt1 = (1-scipy.stats.kendalltau(true_rank, rank1)[0])/2
    kt2 = (1-scipy.stats.kendalltau(true_rank, rank2)[0])/2
    kt3 = (1-scipy.stats.kendalltau(true_rank, rank3)[0])/2
    #print(true_rank, rank1, rank2, rank3)
    return (kt1, kt2, kt3)




def top_k_error(num_paper, noise_variance, k):
    As = find_assignment(num_paper)
    gaussians = gaussian_variables(num_paper, noise_variance)
    (ks, bs, qualities, noises, true_rank) = gaussians
    rank1 = method_1(num_paper, As, gaussians)
    rank2 = method_2(num_paper, As, gaussians)
    rank3 = method_3(num_paper, As, gaussians)
    

    tke1 = top_k_distance(num_paper, true_rank, rank1, k)/k
    tke2 = top_k_distance(num_paper, true_rank, rank2, k)/k
    tke3 = top_k_distance(num_paper, true_rank, rank3, k)/k

    return (tke1, tke2, tke3)



def top_k_distance(num_paper, true_rank, rank, k):
    error = 0
    for i in range(num_paper):
        if true_rank[i] < k and rank[i] >= k:
            error+=1
    return error




def messy_middle_distance(num_paper, true_rank, rank):
    error = 0
    for i in range(num_paper):
        if (9 < true_rank[i]) and (true_rank[i] < 25) and (rank[i] >= 25):
            error+=1
        elif (25 <= true_rank[i]) and (true_rank[i] < 40) and (rank[i] < 25):
            error+=1
    return error



def messy_middle_error(num_paper, noise_variance):
    As = find_assignment(num_paper)
    gaussians = gaussian_variables(num_paper, noise_variance)
    (ks, bs, qualities, noises, true_rank) = gaussians
    rank1 = method_1(num_paper, As, gaussians)
    rank2 = method_2(num_paper, As, gaussians)
    rank3 = method_3(num_paper, As, gaussians)
    

    mme1 = messy_middle_distance(num_paper, true_rank, rank1)/30
    mme2 = messy_middle_distance(num_paper, true_rank, rank2)/30
    mme3 = messy_middle_distance(num_paper, true_rank, rank3)/30

    return (mme1, mme2, mme3)


def simulation(num_paper, noise_variance, k):
    error1 = []
    error2 = []
    error3 = []
    for i in range(100):
        #(tke1, tke2, tke3) = top_k_error(num_paper, noise_variance, k)
        #(tke1, tke2, tke3) = messy_middle_error(num_paper, noise_variance)
        (tke1, tke2, tke3) = kt_distance(num_paper, noise_variance)
        error1.append(tke1)
        error2.append(tke2)
        error3.append(tke3)
    return (sum(error1)/100, sum(error2)/100, sum(error3)/100, np.std(error1), np.std(error2), np.std(error3))



def plot_tke():
    variances = [i/20 for i in range(11)]
    error1 = []
    error2 = []
    error3 = []
    error_bar1 = []
    error_bar2 = []
    error_bar3 = []
    for noise_variance in variances:
        (e1, e2, e3, s1, s2, s3) = simulation(100, noise_variance, 25)
        error1.append(e1)
        error2.append(e2)
        error3.append(e3)
        error_bar1.append(s1/10)
        error_bar2.append(s2/10)
        error_bar3.append(s3/10)
    print(error1)
    print(error2)
    print(error3)
    print(error_bar1)
    print(error_bar2)
    print(error_bar3)

    plt.figure(figsize=(10,7))
    method1, = plt.plot(variances, error1, 's',label = "No calibration", color='r', markersize=16)
    plt.errorbar(variances, error1, yerr = error_bar1, color='r', capsize=6, elinewidth=8, linestyle='--', linewidth=4)

    method2, = plt.plot(variances, error2, '^',label = "Within-conference calibration", color='g', markersize=16)
    plt.errorbar(variances, error2, yerr = error_bar2, color='g', capsize=6, elinewidth=8, linestyle='-.', linewidth=4)

    method3, = plt.plot(variances, error3, 'o',label = "Calibration with known parameters", color='b', markersize=16)
    plt.errorbar(variances, error3, yerr = error_bar3, color='b', capsize=6, elinewidth=8, linestyle='-', linewidth=4)


   
    #plt.ylabel("Average Kendall-tau distance")
    plt.ylabel("Average messy middle error")
    plt.xlabel('Variance of noise')
    plt.legend(prop={'size': 17}, frameon=False)
    

    plt.savefig('messy_middle.pdf', bbox_inches='tight')
    #plt.savefig('kendall_tau.pdf', bbox_inches='tight')

    plt.show()


