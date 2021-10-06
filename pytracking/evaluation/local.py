from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    #settings.network_path = '/home/zikun/work/tracking/KPTracking/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.network_path = '/home/zikun/work/tracking/KPTracking/New_KPT/new_pytracking/checkpoints/ltr/dimp/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/zikun/repository/data/siamrpn_test_data/testing_dataset/OTB2015'
    #settings.otb_path = '/home/zikun/data/OTB2015'
    settings.result_plot_path = '/home/zikun/work/tracking/KPTracking/New_KPT/pytracking_kp_newsa/pytracking/result_plots/'
    settings.results_path = '/home/zikun/work/tracking/KPTracking/New_KPT/pytracking_kp_newsa/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/zikun/work/tracking/KPTracking/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings
