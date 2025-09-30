
def plot_umap_with_images(data, setname, savepath):
    import umap
    from matplo(tlib.offsetbox import OffsetImage, AnnotationBbox

    data = data[0]
    images = data
    reducer = umap.UMAP()
    data = data.reshape(data.shape[0], -1)
    embedding = reducer.fit_transform(data)
    fig, ax = plt.subplots()
    for i in range(embedding.shape[0]):
        xi = embedding[i, 0]
        yi = embedding[i, 1]
        img = images[i]
        imagebox = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(imagebox, (xi, yi), frameon=False)
        ax.add_artist(ab)
    # plt.scatter(
    #     embedding[:, 0],
    #     embedding[:, 1],
    # )  # c=[sns.color_palette()[x] for x in data.classes]
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP projection of {setname}', fontsize=24)
    plt.savefig(savepath)
    plt.clf())


def plot_umap(data, setname, savepath):
    import umap

    data = data[0]
    reducer = umap.UMAP()
    data = data.reshape(data.shape[0], -1)
    embedding = reducer.fit_transform(data)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
    )  # c=[sns.color_palette()[x] for x in data.classes]
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP projection of {setname}', fontsize=24)
    plt.savefig(savepath)
    plt.clf()


def plot_tsne(data, setname, savepath):
    from sklearn.manifold import TSNE

    data = data[0]
    data = data.reshape(data.shape[0], -1)
    tsne = TSNE(n_components=2, learning_rate='auto',
                init='random', perplexity=3)
    embedding = tsne.fit_transform(data)
    plt.scatter(embedding[:, 0],
                embedding[:, 1])
    plt.title(f'TSNE projection of {setname}')
    plt.savefig(savepath)
    plt.clf()
