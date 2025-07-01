import os

# 获取仓库文件列表
def get_repo_files(repoid):
    from modelscope.hub.api import HubApi
    api = HubApi()
    file_paths = []
    try:
        files = api.get_model_files(
            model_id=repoid,
            root=None,
            recursive=True
        )
        
        file_paths = [f['Path'] for f in files if f.get('Type') == 'blob']
        return file_paths
    except Exception as e:
        print(f"❌ 获取仓库 {repoid} 文件列表失败：\n", e)

# 克隆仓库
def download_ms(repoid, dir_path):
    from modelscope.hub.file_download import model_file_download
    wholeness_files = False
    file_paths = get_repo_files(repoid)
    if file_paths:
        for path in file_paths:
            try:
                model_dir = model_file_download(
                    model_id=repoid, file_path=path, local_dir=dir_path
                )
                wholeness_files = True
            except Exception as e:
                print(f"❌ 文件 {path} 下载失败")
    return wholeness_files

# 检测本地文件是否齐全
def check_repo_wholeness(repoid, dir_path):
    absence = False
    if not os.path.exists(dir_path):
        absence = True
    else:
        file_paths = get_repo_files(repoid)
        if file_paths:
            for path in file_paths:
                if not os.path.exists(os.path.join(dir_path, path)):
                    absence = True
    return absence
    