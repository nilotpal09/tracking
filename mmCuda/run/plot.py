import matplotlib.pyplot as plt


# iter_counts = []
# with open('/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/triplet_iter_count_event1.txt') as f:
#     for line in f.readlines():
#         iter_counts.append(int(line))


# with plt.style.context('seaborn'):
#     fig = plt.figure(figsize=(6,6), dpi=200)

#     ax = fig.add_subplot(1,1,1)
#     ax.hist(iter_counts, histtype='stepfilled', color='cornflowerblue', bins=100)
#     ax.hist(iter_counts, histtype='step', color='k', bins=100)
#     ax.set_xlabel('number of iterations (m1 x m2 x m3)')
#     ax.set_ylabel('triplet count')
#     ax.set_yscale('log')
#     ax.set_title('Number of iterations per triplet')

#     text = f'max: {max(iter_counts):,}\nmin: {min(iter_counts):,}\nsum: {sum(iter_counts):,}'

#     ax.text(0.55, 0.95, text,
#             verticalalignment='top', horizontalalignment='left',
#             transform=ax.transAxes,
#             color='k', fontsize=12)

#     plt.savefig('/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/triplet_iter_count_log_event1.png')



#     fig = plt.figure(figsize=(6,6), dpi=200)

#     ax = fig.add_subplot(1,1,1)
#     ax.hist(iter_counts, histtype='stepfilled', color='cornflowerblue', bins=100)
#     ax.hist(iter_counts, histtype='step', color='k', bins=100)
#     ax.set_xlabel('number of iterations (m1 x m2 x m3)')
#     ax.set_ylabel('triplet count')
#     ax.set_title('Number of iterations per triplet')

#     text = f'max: {max(iter_counts):,}\nmin: {min(iter_counts):,}\nsum: {sum(iter_counts):,}'

#     ax.text(0.55, 0.95, text,
#             verticalalignment='top', horizontalalignment='left',
#             transform=ax.transAxes,
#             color='k', fontsize=12)

#     plt.savefig('/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/triplet_iter_count_event1.png')




iter_counts = []
with open('/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/doublet_iter_count_event1.txt') as f:
    for line in f.readlines():
        iter_counts.append(int(line))


with plt.style.context('seaborn'):
    fig = plt.figure(figsize=(6,6), dpi=200)

    ax = fig.add_subplot(1,1,1)
    ax.hist(iter_counts, histtype='stepfilled', color='cornflowerblue', bins=100)
    ax.hist(iter_counts, histtype='step', color='k', bins=100)
    ax.set_xlabel('number of iterations (m1 x m2)')
    ax.set_ylabel('doublet count')
    ax.set_yscale('log')
    ax.set_title('Number of iterations per doublet')

    text = f'max: {max(iter_counts):,}\nmin: {min(iter_counts):,}\nsum: {sum(iter_counts):,}'

    ax.text(0.55, 0.95, text,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='k', fontsize=12)

    plt.savefig('/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/doublet_iter_count_log_event1.png')



    fig = plt.figure(figsize=(6,6), dpi=200)

    ax = fig.add_subplot(1,1,1)
    ax.hist(iter_counts, histtype='stepfilled', color='cornflowerblue', bins=100)
    ax.hist(iter_counts, histtype='step', color='k', bins=100)
    ax.set_xlabel('number of iterations (m1 x m2)')
    ax.set_ylabel('doublet count')
    ax.set_title('Number of iterations per doublet')

    text = f'max: {max(iter_counts):,}\nmin: {min(iter_counts):,}\nsum: {sum(iter_counts):,}'

    ax.text(0.55, 0.95, text,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes,
            color='k', fontsize=12)

    plt.savefig('/srv01/agrp/nilotpal/projects/tracking/mmCuda/run/doublet_iter_count_event1.png')

