'''
script for the main plots of the paper
author: harry.coppock@imperial.ac.uk
'''
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.transforms as mtransforms
import geopandas

from geomap import plot_geo
from upset_plots import upset
from sub_date import sub_time_by_cov

def figure_1():
    '''
    Plot the main figure for the paper
    '''
    # load the uk county boundaries
    url = 'https://opendata.arcgis.com/api/v3/datasets/83f458a118604169b599000411f364bf_0/downloads/data?format=shp&spatialRefId=27700'
    data = geopandas.read_file(url)
    #main figure
    fig = plt.figure(figsize=(15,5))
    # first the geoplot
    ax1 = plt.subplot(151)
    plot_geo(data, 'Positive', ax1)
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.26, 0.2, 0.01, 0.6])
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax1.text(0.0, 1.0, 'a) Positive', transform=ax1.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0, alpha=0))

    fig.colorbar(im, cax=cax)
    ax2 = plt.subplot(152)
    plot_geo(data, 'Negative', ax2)
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.42, 0.2, 0.01, 0.6])
    fig.colorbar(im, cax=cax)
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax2.text(0.0, 1.0, 'b) Negative', transform=ax2.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0, alpha=0))
    ax3 = plt.subplot(153)
    plot_geo(data, 'Negative - Positive', ax3)
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.585, 0.2, 0.01, 0.6])
    fig.colorbar(im, cax=cax)
    cax.set_ylabel('Difference in % proportion')
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax3.text(0.0, 1.0, 'c) Negative - Positive', transform=ax3.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0, alpha=0))
    
    ax6 = plt.subplot(144)
    sub_time_by_cov(ax6)
    trans = mtransforms.ScaledTranslation(-20/72, -5/72, fig.dpi_scale_trans)
    ax6.text(0.0, 1.0, 'd)', transform=ax6.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0, alpha=0))
    
    #upset('Positive')
    #upset('Negative')
    
    #ax4 = plt.subplot(223)
    #positive_upset = image.imread('figs/Positive_upset.png')
    #ax4.imshow(positive_upset, aspect='equal')
    #ax4.axis('off')
    #ax5 = plt.subplot(224)
    #negative_upset = image.imread('figs/Negative_upset.png')
    #ax5.imshow(negative_upset, aspect='equal')
    #ax5.axis('off')



    #plt.tight_layout()

    plt.savefig(f'figs/figure1_opt1.png', bbox_inches='tight')

def figure_1_sub_1(data):
    fig, ax = plt.subplots()
    plot_geo(data, 'Positive', ax)
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.7, 0.2, 0.02, 0.7])
    fig.colorbar(im, cax=cax)
    plt.savefig('figs/positive_geo.png', bbox_inches='tight')
    plt.close()

def figure_1_sub_2(data):
    fig, ax = plt.subplots()
    plot_geo(data, 'Negative', ax)
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.7, 0.2, 0.02, 0.7])
    fig.colorbar(im, cax=cax)
    plt.savefig('figs/negative_geo.png', bbox_inches='tight')
    plt.close()

def figure_1_sub_3(data):
    fig, ax = plt.subplots()
    plot_geo(data, 'Negative - Positive', ax)
    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.7, 0.2, 0.02, 0.7])
    fig.colorbar(im, cax=cax)
    cax.set_ylabel('Difference in % proportion')
    plt.savefig('figs/dif_geo.png', bbox_inches='tight')
    plt.close()

def figure_1_sub_4():
    fig, ax = plt.subplots()
    sub_time_by_cov(ax)
    plt.savefig('figs/sub_time.png', bbox_inches='tight')
    plt.close()


def figure_1_subs():
    url = 'https://opendata.arcgis.com/api/v3/datasets/83f458a118604169b599000411f364bf_0/downloads/data?format=shp&spatialRefId=27700'
    data = geopandas.read_file(url)
    figure_1_sub_1(data)
    figure_1_sub_2(data)
    figure_1_sub_3(data)
    figure_1_sub_4()
    upset('Positive')
    upset('Negative')

if __name__ == '__main__':
    figure_1()
