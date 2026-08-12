#https://rstudio.github.io/cheatsheets/html/shiny-python.html
#https://shiny.posit.co/py/docs/overview.html
#https://shiny.posit.co/py/docs/install-create-run.html

#https://shiny.posit.co/py/components/outputs/plot-plotly/
#https://shiny.posit.co/py/components/

#https://shiny.posit.co/py/api/core/run_app.html
#https://github.com/posit-dev/py-shinyswatch

from typing import List

import nltk
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime


from shiny import App, Inputs, Outputs, Session, reactive, ui, render
from shiny.express import ui as ui_express

from shiny.types import NavSetArg
from shiny.types import ImgData
from pathlib import Path
from shinywidgets import output_widget, render_widget
#import shinyswatch

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from collections import Counter , OrderedDict

import plotly.express as px
import plotly.graph_objects as go

#For text preprocessing
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


from shiny import run_app

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import bigrams

import base64
import pandas as pd
from tomotopy.utils import Corpus

import tomotopy as tp
import pyLDAvis

import math
from pyvis.network import Network


#https://scikit-learn.org/stable/modules/clustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering , DBSCAN

import nimfa

from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor



from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.decomposition import PCA


import statsmodels.api as sm

from numpy import array, random, arange
from itertools import chain




'''
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

'''


def text_preprocessing_fun(docs_df,txt_preprocessing_choices):
    #"lowercase_option": "Lowercase transformation",
    #"html_option": "HTML removal",
    #"punctuation_option": "Punctuation removal",
    #"stopwords_option": "Stopwords removal",
    #"lemmatize_option": "Lemmatize tokens"
    #"numbers_option": "Remove numbers"
    #"bigrams_option": "Create bigrams"
    #"stemmer_option": "Stem tokens"

    docs_df['new_text'] = docs_df['new_text'].apply(lambda temp: str(temp))

    def remove_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    if "html_option" in txt_preprocessing_choices:
        docs_df['new_text'] = docs_df['new_text'].apply(remove_html_tags)

    if "lowercase_option" in txt_preprocessing_choices:
        docs_df['new_text'] = docs_df['new_text'].str.lower()


    if "punctuation_option" in txt_preprocessing_choices:
        def remove_punctuation(text):
            return text.translate(str.maketrans('', '', string.punctuation))
        docs_df['new_text'] = docs_df['new_text'].apply(remove_punctuation)


    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        return ' '.join(word for word in text.split() if word not in stop_words)

    if "stopwords_option" in txt_preprocessing_choices:
        docs_df['new_text'] = docs_df['new_text'].apply(remove_stopwords)


    def remove_numbers(text):
        return re.sub(r'\d+', '', text)

    if "numbers_option" in txt_preprocessing_choices:
        docs_df['new_text'] = docs_df['new_text'].apply(remove_numbers)


    if "lemmatize_option" in txt_preprocessing_choices:
        lemmatizer = WordNetLemmatizer()

        def lemmatize_words(text):
            return ' '.join(lemmatizer.lemmatize(word) for word in text.split())
        docs_df['new_text'] = docs_df['new_text'].apply(lemmatize_words)
    if "stemmer_option" in txt_preprocessing_choices:
        stemmer = PorterStemmer()

        def stemfun(text):
            return ' '.join(stemmer.stem(word) for word in text.split())

        docs_df['new_text'] = docs_df['new_text'].apply(stemfun)


    if "bigrams_option" in txt_preprocessing_choices:
        def bigrams_words(text):
            temp = word_tokenize(text)
            temp = list(bigrams(temp))
            return ' '.join(['_'.join(pair) for pair in temp])

        docs_df['new_text'] = docs_df['new_text'].apply(bigrams_words)

    return(docs_df)


def xicor(X, Y, ties=True):

    random.seed(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                r[tie_index] = random.choice(r[tie_index] - arange(0, sum([r[j] == r[i] for i in range(n)])), sum(tie_index), replace=False)
        return (1 - n*sum( abs(r[1:] - r[:n-1]) ) / (2*sum(l*(n - l))))
    else:
        r = array([sum(y >= Y[order]) for y in Y[order]])
        return (1 - 3 * sum( abs(r[1:] - r[:n-1]) ) / (n**2 - 1))





def main_nav_controls() -> List[NavSetArg]:
    return [

        ui.nav_panel("Home",
                     ui.card(
                         ui.HTML("<h1><strong> Load file </strong> </h1>"),
                        ui.input_file("load_file","",accept=[".csv"]),
                         ui.row(ui.column(2, ui.card(ui.output_text("no_posts_head"))),
                                ui.column(2, ui.card(ui.output_text("no_tags_head"))),
                                ui.column(2, ui.card(ui.output_text("no_users_head"))),
                                ui.column(2, ui.card(ui.output_text("no_views_head"))),
                                ui.column(2, ui.card(ui.output_text("no_answers_head"))),
                                ui.column(2, ui.card(ui.output_text("no_comments_head"))),
                                ),
                                                               ui.output_data_frame("main_data_table"),

                     ),

                     ),
        ui.nav_panel("Statistics",
                     ui.navset_tab(
                    ui.nav_panel("Posts",

                                 ui.card(
                                     ui.HTML("<h1><strong> Questions posted in time  </strong></h1>"),
                                     ui.input_radio_buttons("month_year_posts_sel","Per year or month",choices={"year_opt":"Years","month_opt":"Months"}),
                                     output_widget("month_year_posts"),
                                     ui.span(),
                                 ),

                                ui.card(
                                    ui.HTML("<h1><strong> Barplot of Views  </strong></h1>"),

                                    ui_express.input_numeric("barplot_bins_views","Number of bins",1000,min=1,max=2000),
                                    output_widget("barplot_view_count"),
                                    ui.span(),

                                    ui.HTML("<h1><strong> Barplot of Answers  </strong></h1>"),

                                    ui_express.input_numeric("barplot_bins_answers", "Number of bins", 100, min=1, max=200),
                                    output_widget("barplot_answer_count"),
                                    ui.span(),

                                    ui.HTML("<h1><strong> Barplot of Comments  </strong></h1>"),

                                    ui_express.input_numeric("barplot_bins_comments", "Number of bins", 100, min=1, max=200),
                                    output_widget("barplot_comment_count"),
                                    ui.span(),

                                    ui.HTML("<h1><strong> Barplot of Scores  </strong></h1>"),

                                    ui_express.input_numeric("barplot_bins_score", "Number of bins", 1000, min=1, max=200),
                                    output_widget("barplot_score_count"),

                                    #ui.output_plot("barplot_view_count")
                                 ),

                                  ),
                    ui.nav_panel("Tags",


                                 ui.card(
                                     ui.HTML("<h1><strong>  Tag Statistics  </strong></h1>"),
                                     ui.span(),
                                 ui.row(ui.column(4,ui.output_data_frame("tag_frequency_table")),ui.column(4,ui.output_plot("tag_wordcloud_output")))

                                 ),

                                 ),
                     )




                     ),
        ui.nav_panel("Text preprocessing",
                     ui.card(
                ui.HTML("<h1> <strong> Settings </strong> </h1>"),
                ui.input_radio_buttons(
                    "text_content_options",
                    "Text content options",
                    {"text_only": "Title Only", "body_only":"Body Only","text_body":"Title and Body"}
                )
                     ,

                ui.input_checkbox_group("txt_preprocessing_choices",
                                        "Text preprocessing options",
                                        {
                    "lowercase_option": "Lowercase transformation",
                    "html_option": "HTML removal",
                    "punctuation_option": "Punctuation removal",
                    "stopwords_option": "Stopwords removal",
                    "numbers_option":"Remove numbers",
                    "lemmatize_option": "Lemmatize tokens",
                    "stemmer_option": "Stem tokens",
                    "bigrams_option": "Create bigrams"
                                        }
                                        ),
                         ui.span(),

                         ui.input_action_button("txt_preprocessing_button", "Start text preprocessing"),
                     ),


                     ui.span(),


                ui.card(
                    ui.HTML("<h1><strong>  Text Statistics  </strong></h1>"),
                    ui.row(ui.column(4,ui.output_data_frame("word_frequency_table")),ui.column(4,ui.output_plot("word_wordcloud_output")))  ,


                ),
                ),
        ui.nav_panel("Topic modeling",
                     ui.navset_tab(
                         ui.nav_panel("Model",

                                      ui.card(
                                          ui.HTML("<h1><strong> Topic model options and settings </strong> </h1>"),
                                          ui_express.input_radio_buttons(
                                              "topic_model_option", "Select topic modeling algorithm",
                                              {"lda_model_option": "Latent Dirichlet Allocation (LDA)",
                                               "ctm_model_option": "Correlated Topic Models (CTM)",
                                               "ptm_model_option": "Pseudo-document based Topic Model (PTM)",
                                               "dmr_model_option": "Dirichlet Multinomial Regression (DMR)"
                                               }
                                          ),
                                          ui.input_radio_buttons(id="topic_model_term_weighting", label="Term weigthing",
                                                             choices={"bin_topic_model_term_weighting": "Binary",
                                                                      "idf_topic_model_term_weighting": "Inverse Document Frequency (IDF)",
                                                                      "pmi_topic_model_term_weighting": "Pointwise Mutual Information (PMI)"}
                                                             ),
                                          ui_express.input_numeric("no_topics_option", "Number of topics", 10, min=2,
                                                                   max=200),
                                          ui_express.input_numeric("no_iterations_topic_model", "Training iterations",
                                                                   10, min=1, max=10000),

                                          ui.row(ui.column(2, ui.input_numeric("min_df_opt",label="Minimum Document Frequency of words",value=10,min=2,max=10000)),
                                                 ui.column(2, ui.input_numeric("rm_top_opt",label="Number of most frequent words to exclude",value=0,min=0,max=10000)),
                                                 ),
                                          ui.span(),
                                          ui.input_action_button("train_topic_model_button", "Train topic model"),

                                      ),



                                      ui.span(),

                                      ui.card(
                                          ui.HTML("<h1><strong> Outputs </strong> </h1>"),
                                          ui.card(
                                          ui.HTML("<h2><strong> Topic visualization </strong> </h2>"),
                                          ui.output_ui("lda_vis_topic_model")
                                          ),
                                          ui.card(
                                              ui.HTML("<h2><strong> Topic evaluation </strong> </h2>"),
                                              ui.input_numeric(id='no_topic_top_words',label="Number of Top words per topic",value=10,min=2,max=10000),
                                              ui.row(ui.column(4, ui.card(ui.output_text("topic_coherence_output_text"))),
                                                     ui.column(4, ui.card(ui.output_text("topic_divergence_output_text"))),
                                                     ui.column(4, ui.card(ui.output_text("topic_cumscore_output_text")))
                                                     )
                                          ),
                                          ui.card(
                                          ui.HTML("<h2><strong> Top10 documents per topic </strong> </h2>"),
                                          ui.input_select(id="topdocs_topic_opt",label="Select topic",choices={})
                                          ),
                                          ui.input_action_button(id="topdocs_topic_button",label="Present top documents"),
                                          ui.column(12,ui.output_table(id="topdocs_topic_vis_table"))
                                      )
                                      )
                         ,
                         ui.nav_panel("Topic Popularity and difficulty",
                                      ui.card(
                                      ui.HTML("<h1><strong> Topic popularity and difficulty </strong> </h1>"),
                                      ui.input_radio_buttons(
                                          "pop_dif_weight_option",
                                          "Topic properties",
                                          {"max_weight": "Only dominant topic", "all_weight": "Use weights","corr_option":"Spearman correlation"}
                                      ),
                                      ui.input_action_button("pop_dif_metrics_button_topic", "Calculate analytics"),
                                      ),
                                      ui.card(

                                          ui.column(10, ui.HTML("<h2><strong> Popularity Table </strong> </h2>"),
                                                    ui.output_table("pop_metrics_table_topic")),
                                          # ui.output_data_frame("pop_metrics_table_topic")

                                          ui.column(10, ui.HTML("<h2><strong> Difficulty Table </strong> </h2>"),
                                                    ui.output_table("dif_metrics_table_topic")),

                                          ui.HTML("<h2><strong> 2d Matrix with metrics </strong> </h2>"),
                                          ui.card(
                                          ui.row(ui.column(4,ui.input_radio_buttons(id="topic_2dmetrics_1",label="Metric 1"
                                            ,choices={"views_opt":"Average Views","score_opt":"Weight Score","comm_opt":"Weight Comments","answer_opt":"Weight Answers","answer_views_opt":"Weight Answers / Weight Views","acc_opt":"Weight with accepted answers","poppc_opt":"Popularity Principal component","difpc_opt":"Difficulty Principal component"}))
                                                 ,ui.column(4,ui.input_radio_buttons(id="topic_2dmetrics_2",label="Metric 2"
                                            ,choices={"views_opt":"Average Views","score_opt":"Weight Score","comm_opt":"Weight Comments","answer_opt":"Weight Answers","answer_views_opt":"Weight Answers / Weight Views","acc_opt":"Weight with accepted answers","poppc_opt":"Popularity Principal component","difpc_opt":"Difficulty Principal component"})),

                                                 )
                                          ),
                                      ui.input_action_button("topic2d_metric_button","Complete 2d Visualization of topic metrics"),
                                      output_widget("topic2d_vis_widget")

                                      ),
                                      ),
                         ui.nav_panel("Topic Growth",
                                      ui.card(
                                          ui.HTML("<h2><strong> Topic properties </strong> </h2>"),

                                          ui.input_radio_buttons(
                                          "topic_growth_weight_option",
                                          "",
                                          {"max_weight": "Only dominant topic", "all_weight": "Use weights"}
                                      ),
                                      ui.input_action_button("growth_metrics_button_topic", "Calculate analytics"),
                                      ),

                                      ui.card(
                                          ui.HTML("<h1><strong> Topic Growth accross years </strong> </h1>"),
                                      ui.input_radio_buttons(id="topic_model_growth_my",label="Year or Month",
                                                         choices={"per_year":"Year","per_month":"Month"}),
                                      output_widget("topic_growth_year_vis")
                                      ),

                                      ),
                         ui.nav_panel("Regression modeling",

                                    ui.card(
                                        ui.HTML("<h1><strong> Regression models </strong> </h1>"),
                                        ui.input_radio_buttons("topic_reg_weight",
                                                               "Weight option",
                                                               {"weight_opt":"Topic frequency","prop_opt":"Topic proportions"}
                                        ),
                                      ui.input_radio_buttons(
                                          "topic_reg_opt",
                                          "Model options",
                                          { "grad_opt":"Gradient Boosting","dec_tree_opt":"Decision Trees","rf_opt": "Random Forest","linear_opt": "Linear Regression",
                                            "pois_opt":"Poisson Regression",
                                            "zero_infl_poiss":"Zero Inflated Poisson Regression",
                                            "neg_bin_opt":"Negative Binomial Regression",
                                            "zero_neg_bin_opt":"Zero Inflated Negative Binomial Regression",
                                            "bin_opt":"Binomial Regression"
                                            }
                                      ),
                                      ui.input_radio_buttons(
                                          "topic_reg_output",
                                          "Model options",
                                          {"views_opt": "Views", "score_opt": "Score", "comm_opt":"Comments",
                                           "answer_opt": "Answers", "ans_view_opt":"Answers / Views","acc_opt":"Has accepted answer",
                                           "year_opt":"Year", "time_opt":"Timestamp"

                                           }
                                      ),
                                    ui.input_action_button("topic_reg_button", "Train model"),
                                    ),



                                      ui.card(
                                          ui.HTML("<h1><strong> Output </strong> </h1>"),
                                          ui.column(8,ui.output_table("topic_reg_table"))
                                      )

                                      )


                     ),



                     ),
        ui.nav_panel("Tag clustering",
                        ui.navset_tab(ui.nav_panel("Tag clusters",
                                                ui.card(
                                                   ui.card(
                                                       ui.HTML("<h1><strong> Exclude Tags </strong> </h1>"),
                                                       ui.row(ui.column(2, ui_express.input_numeric(
                                                           id="low_thres_tag_clust",
                                                           label="Minimum percentage of questions containing the tag",
                                                           value=5, min=0, max=100)),
                                                              ui.column(2,
                                                                        ui_express.input_numeric("up_thres_tag_clust",
                                                                                                 "Maximum percentage of questions containing the tag",
                                                                                                 100, min=0, max=100))
                                                              ),
                                                       ui.input_text_area(id="exclude_tag_text",
                                                                          label="Exclude tags manually",
                                                                          value="",
                                                                          placeholder="Separate multiple tags using an empty space",
                                                                          resize="both")

                                                   ),


                                                   ui.card(
                                                       ui.HTML("<h1><strong> Clustering options </strong> </h1>"),
                                                       ui.row(
                                                           ui.column(4, ui.input_radio_buttons(
                                                               id="tag_clust_options_alg",
                                                               label="Clustering algorithm",
                                                               choices={"affinity_clust_option": "Affinity Propagation Clustering",
                                                                        "spectral_clust_option": "Spectral Clustering",
                                                                        "dbscan_clust_option": "DBSCAN Clustering"}
                                                           )),
                                                           ui.column(4, ui.input_radio_buttons(
                                                               id="tag_clust_options_weight",
                                                               label="Tag similarity function",
                                                               choices={"ii_weight_option": "Inclusion Index",
                                                                        "ri_weight_option": "Reverse Inclusion Index",
                                                                        "ji_weight_option": "Jaccard Similarity Index",
                                                                        "ei_weight_option": "Equivalence Index"
                                                                        }
                                                           )),
                                                           ui.column(4,ui.panel_conditional("input.tag_clust_options_alg === 'spectral_clust_option' || input.tag_clust_options_alg === 'dbscan_clust_option'",ui.input_numeric("extra_opt_clust","Number of Clusters",value=10,min=2,max=200)))
                                                       ),

                                                   ),
                                                    ui.input_action_button("tag_clust_button",
                                                                           "Start Tag Clustering"),
                                                ),

                                                ui.card(
                                                    ui.HTML("<h1><strong> Outputs </strong> </h1>"),
                                                   ui.card(

                                                       ui.HTML("<h2><strong> Cluster details </strong> </h2>"),
                                                       ui.input_select(id="cluster_no_sel",label="Select a cluster to inspect",choices={}),
                                                       ui.output_data_frame("cluster_top_tags")
                                                           ),

                                                ui.panel_conditional("'TRUE'==='FALSE'",
                                                   ui.card( ui.HTML("<h2><strong> Cluster visualization </strong> </h2>"),
                                                            ui.input_action_button("show_tag_clust_button","Show tag clustering visualization"),


                                                       ui.output_ui("tag_clust_vis")
                                                   )
                                                )
                                                   )
                                                   ),
                                      ui.nav_panel("Tag and Tag cluster Popularity and Difficulty",

                                      ui.card(
                                      ui.HTML(
                                          "<h1><strong> Tag cluster popularity and difficulty </strong> </h1>"),
                                                   ui.input_radio_buttons(id="pop_dif_weight_option_clust",
                                                                      label="Select a weighting strategy",
                                                                      choices={"max_weight": "Only dominant cluster","all_weight":"Weighting option","corr_option":"Spearman correlation"}),

                                                   ui.input_action_button("pop_dif_metrics_button_tag_clust", "Calculate analytics"),
                                      ),


                                      ui.card(
                                          ui.column(10, ui.HTML("<h2><strong> Popularity Table </strong> </h2>"),
                                                    ui.output_table("pop_metrics_table_tag_clust")),
                                          # ui.output_data_frame("pop_metrics_table_topic")

                                          ui.column(10, ui.HTML("<h2><strong> Difficulty Table </strong> </h2>"),
                                                    ui.output_table("dif_metrics_table_tag_clust")),

                                          ui.HTML("<h2><strong> 2d Matrix with metrics </strong> </h2>"),
                                          ui.card(
                                              ui.row(ui.column(4, ui.input_radio_buttons(id="tag_clust_2dmetrics_1",
                                                                                         label="Metric 1"
                                                                                         , choices={"views_opt":"Average Views","score_opt":"Weight Score","comm_opt":"Weight Comments","answer_opt":"Weight Answers","answer_views_opt":"Weight Answers / Weight Views","acc_opt":"Weight with accepted answers","poppc_opt":"Popularity Principal component","difpc_opt":"Difficulty Principal component"}))
                                                     , ui.column(4, ui.input_radio_buttons(id="tag_clust_2dmetrics_2",
                                                                                           label="Metric 2"
                                                                                         , choices={"views_opt":"Average Views","score_opt":"Weight Score","comm_opt":"Weight Comments","answer_opt":"Weight Answers","answer_views_opt":"Weight Answers / Weight Views","acc_opt":"Weight with accepted answers","poppc_opt":"Popularity Principal component","difpc_opt":"Difficulty Principal component"})),

                                                     )
                                          ),
                                          ui.input_action_button("tag_clust2d_metric_button",
                                                                 "Complete 2d Visualization of topic metrics"),
                                          output_widget("tag_clust_vis_widget")
                                      )

                                      ),
                                      ui.nav_panel("Cluster Growth",
                                                    ui.card(
                                                   ui.HTML("<h1><strong> Tag clusters Growth </strong> </h1>"),
                                                   ui.input_radio_buttons(
                                                       "cluster_growth_weight_option",
                                                       "Cluster properties",
                                                       {"max_weight": "Only dominant topic",
                                                        "all_weight": "Use weights"}
                                                   ),
                                                   ui.input_action_button("growth_metrics_button_cluster",
                                                                          "Calculate analytics"),
                                                    ),
                                                   ui.card(
                                                       ui.HTML("<h2><strong> Year or Month </strong> </h2>"),

                                                       ui.input_radio_buttons(id="cluster_model_growth_my",
                                                                              label="",
                                                                              choices={"per_year": "Year",
                                                                                       "per_month": "Month"}),
                                                       output_widget("cluster_growth_year_vis")
                                                   )

                                                   ),

                                      ui.nav_panel("Regression modeling",

                                                   ui.card(
                                                       ui.HTML("<h1><strong> Regression models </strong> </h1>"),
                                                       ui.input_radio_buttons(
                                                           "clust_reg_opt",
                                                           "Model options",
                                                           {"grad_opt": "Gradient Boosting",
                                                            "dec_tree_opt": "Decision Trees", "rf_opt": "Random Forest",
                                                            "linear_opt": "Linear Regression",
                                                            "pois_opt": "Poisson Regression",
                                                            "zero_infl_poiss": "Zero Inflated poisson Regression",
                                                            "neg_bin_opt": "Negative Binomial Regression",
                                                            "zero_neg_bin_opt": "Zero Inflated Negative Binomial Regression",
                                                            "bin_opt": "Binomial Regression"
                                                            }
                                                       ),
                                                       ui.input_radio_buttons(
                                                           "clust_reg_output",
                                                           "Model options",
                                                           {"views_opt": "Views", "score_opt": "Score",
                                                            "comm_opt": "Comments",
                                                            "answer_opt": "Answers", "ans_view_opt": "Answers / Views",
                                                            "acc_opt": "Has accepted answer",
                                                                       "year_opt":"Year", "time_opt": "Timestamp"
                                                            }
                                                       ),
                                                       ui.input_action_button("clust_reg_button", "Train model"),
                                                   ),



                                                   ui.card(
                                                       ui.HTML("<h1><strong> Output </strong> </h1>"),
                                                       ui.column(8, ui.output_table("clust_reg_table"))
                                                   )

                                                   )
                     )


                     ),
        ui.nav_panel("Document Clustering",
                     ui.navset_tab(
                         ui.nav_panel("Document clusters",
                                      ui.card(
                                          ui.HTML("<h1><strong> Document clustering model options and settings </strong> </h1>"),
                                                ui.input_radio_buttons("doc_cluster_choices",
                                                                        "Document Clustering options",
                                                                        {
                                                                            "nmf_option": "Standard Nonnegative Matrix Factorization (NMF)",
                                                                            "bmf_option": "Binary Matrix Factorization (BMF)"#,
                                                                            #"pmf_option": "Probabilistic Nonnegative Matrix Factorization (PMF)",
                                                                            #"psmf_option": "Probabilistic Sparse Matrix Factorization (PSMF)"
                                                                        }
                                                                        ),

                                                ui.input_radio_buttons(
                                                    "doc_cluster_feature_options",
                                                    "Content options",
                                                    {"text_only": "Text Only", "tags_only": "Tags Only",
                                                     "text_tags": "Text and Tags"}
                                                ),
                                                ui.input_radio_buttons(
                                              "doc_clust_is_bin",
                                              "Weight option",
                                              {"bin_option": "Binary weighting", "bow_option": "Raw frequency"}
                                                ),

                                                ui_express.input_numeric("no_features_option", "Number of Clusters", 10,
                                                                         min=2,
                                                                         max=200),
                                          ui.row(ui.column(2, ui_express.input_numeric(
                                              id="low_thres_doc_clust",
                                              label="Minimum percentage of questions containing the token",
                                              value=5, min=0, max=100)),
                                                 ui.column(2,
                                                           ui_express.input_numeric("up_thres_doc_clust",
                                                                                    "Maximum percentage of questions containing the token",
                                                                                    100, min=0, max=100))
                                                 ),
                                                ui_express.input_numeric("no_iterations_doc_cluster",
                                                                         "Training iterations",
                                                                         10, min=1, max=10000),
                                                ui.span(),
                                                ui.input_action_button("train_doc_cluster_button", "Train model"),
                                      ),
                                      ui.card(
                                          ui.HTML("<h1><strong> Outputs </strong> </h1>"),
                                          ui.HTML("<h2><strong> Cluster details </strong> </h2>"),
                                          ui.input_select(id="doc_cluster_no_sel", label="Select a cluster to inspect",
                                                             choices={}),
                                          ui.output_data_frame("doc_cluster_top_feat")
                                      ),

                                                ),

                     ui.nav_panel("Cluster Popularity and difficulty",
                                  ui.card(
                                      ui.HTML("<h1><strong> Cluster popularity and difficulty </strong> </h1>"),
                                      ui.input_radio_buttons(
                                          "pop_dif_weight_option_doc",
                                          "Topic properties",
                                          {"max_weight": "Only dominant cluster", "all_weight": "Use weights",
                                           "corr_option": "Spearman correlation"}
                                      ),
                                      ui.input_action_button("pop_dif_metrics_button_doc", "Calculate analytics"),
                                  ),
                                  ui.card(

                                      ui.column(10, ui.HTML("<h2><strong> Popularity Table </strong> </h2>"),
                                                ui.output_table("pop_metrics_table_doc")),
                                      # ui.output_data_frame("pop_metrics_table_topic")

                                      ui.column(10, ui.HTML("<h2><strong> Difficulty Table </strong> </h2>"),
                                                ui.output_table("dif_metrics_table_doc")),

                                      ui.HTML("<h2><strong> 2d Matrix with metrics </strong> </h2>"),
                                      ui.card(
                                          ui.row(ui.column(4, ui.input_radio_buttons(id="doc_2dmetrics_1",
                                                                                     label="Metric 1"
                                                                                     , choices={
                                                  "views_opt": "Average Views", "score_opt": "Weight Score",
                                                  "comm_opt": "Weight Comments", "answer_opt": "Weight Answers",
                                                  "answer_views_opt": "Weight Answers / Weight Views",
                                                  "acc_opt": "Weight with accepted answers",
                                                  "poppc_opt": "Popularity Principal component",
                                                  "difpc_opt": "Difficulty Principal component"}))
                                                 , ui.column(4, ui.input_radio_buttons(id="doc_2dmetrics_2",
                                                                                       label="Metric 2"
                                                                                       , choices={
                                                      "views_opt": "Average Views", "score_opt": "Weight Score",
                                                      "comm_opt": "Weight Comments", "answer_opt": "Weight Answers",
                                                      "answer_views_opt": "Weight Answers / Weight Views",
                                                      "acc_opt": "Weight with accepted answers",
                                                      "poppc_opt": "Popularity Principal component",
                                                      "difpc_opt": "Difficulty Principal component"})),

                                                 )
                                      ),
                                      ui.input_action_button("doc2d_metric_button",
                                                             "Complete 2d Visualization of topic metrics"),
                                      output_widget("doc2d_vis_widget")

                                  ),
                                  ),
                     ui.nav_panel("Cluster Growth",
                                  ui.card(
                                      ui.HTML("<h2><strong> Cluster properties </strong> </h2>"),

                                      ui.input_radio_buttons(
                                          "doc_growth_weight_option",
                                          "",
                                          {"max_weight": "Only dominant cluster", "all_weight": "Use weights"}
                                      ),
                                      ui.input_action_button("growth_metrics_button_doc", "Calculate analytics"),
                                  ),

                                  ui.card(
                                      ui.HTML("<h1><strong> Cluster Growth accross years </strong> </h1>"),
                                      ui.input_radio_buttons(id="doc_model_growth_my", label="Year or Month",
                                                             choices={"per_year": "Year", "per_month": "Month"}),
                                      output_widget("doc_growth_year_vis")
                                  ),

                                  ),
                     ui.nav_panel("Regression modeling",

                                  ui.card(
                                      ui.HTML("<h1><strong> Regression models </strong> </h1>"),
                                      ui.input_radio_buttons(
                                          "doc_reg_opt",
                                          "Model options",
                                          {"grad_opt": "Gradient Boosting", "dec_tree_opt": "Decision Trees",
                                           "rf_opt": "Random Forest", "linear_opt": "Linear Regression",
                                           "pois_opt": "Poisson Regression",
                                           "zero_infl_poiss":"Zero Inflated Poisson Regression",
                                           "neg_bin_opt": "Negative Binomial Regression",
                                           "zero_neg_bin_opt":"Zero Inflated Negative Binomial Regression",
                                           "bin_opt": "Binomial Regression"
                                           }
                                      ),
                                      ui.input_radio_buttons(
                                          "doc_reg_output",
                                          "Model options",
                                          {"views_opt": "Views", "score_opt": "Score", "comm_opt": "Comments",
                                           "answer_opt": "Answers", "ans_view_opt": "Answers / Views",
                                           "acc_opt": "Has accepted answer",
                                           "year_opt": "Year", "time_opt": "Timestamp"

                                           }
                                      ),
                                      ui.input_action_button("doc_reg_button", "Train model"),
                                  ),

                                  ui.card(
                                      ui.HTML("<h1><strong> Output </strong> </h1>"),
                                      ui.column(8, ui.output_table("doc_reg_table"))
                                  )

                                  )


                     )
                     ),
        ui.nav_panel("Outputs"

        ),

        ui.nav_spacer(),
        ui.nav_menu(
            "Links",
            ui.nav_control(
                ui.a(
                    "Shiny",
                    href="https://shiny.posit.co/py/",
                    target="_blank",
                )
            ),

            ui.nav_control(
                ui.a(
                    "Posit",
                    href="https://posit.co",
                    target="_blank",
                )
            ),
            align="right",
        ),
    ]



shiny_ui = ui.page_fluid(#shinyswatch.theme.zephyr,
    ui.output_image(id="main_image_head",height="80px"),

    ui.tags.head(
    ui.tags.style(
        """
            
            body {
            background: linear-gradient(to right, #A2A1FF, #FFFFFF,#FFFFFF,#FFFFFF,#FFFFFF);
        }
                .card {
            background: linear-gradient(to right, #A2A1FF, #FF9C9E);
        }

        .navbar{
        background: linear-gradient(to right, #A2A1FF, #FF9C9E);
        }
        """
    )
),


    ui.page_navbar(

     *main_nav_controls(),
    title="",
    id="navbar_id",
    footer=ui.div(
        {"style": "width:80%;margin: 0 auto"},
        ui.tags.style(
            """
            h4 {
                margin-top: 3em;
            }
            """
        ),

    )
)
)

'''
    ui.tags.head(
    ui.tags.style(
        """
            
            body {
            background: linear-gradient(to right, #FFD0D0, #FFFFFF,#FFFFFF,#FFFFFF,#FFFFFF);
        }
                .card {
            background: linear-gradient(to right, #FFD0D0, #FF9C9E);
        }

        .navbar{
        background: linear-gradient(to right, #FFD0D0, #FF9C9E);
        }
        """
    )
),
'''


def shiny_server(input: Inputs, output: Outputs, session: Session):


    main_data=reactive.value()



    doc_to_tag_matrix=reactive.value()
    tag_to_tag_matrix=reactive.value()
    unique_tags =reactive.value()




    topic_model=reactive.value()
    zero_indexes_list=reactive.value()
    topic_pop_metrics_react=reactive.value()
    topic_dif_metrics_react=reactive.value()
    topic_growth_react=reactive.value()
    topic_growth_month_react=reactive.value()

    model_ap=reactive.value()

    model_doc=reactive.value()
    no_clust_doc=reactive.value()
    feat_doc=reactive.value()
    doc_pop_metrics_react=reactive.value()
    doc_dif_metrics_react=reactive.value()
    doc_growth_react=reactive.value()
    doc_growth_month_react=reactive.value()
    zero_indexes_list_docs=reactive.value()

    def affinity_prop_model_vis(tag_to_tag_matrix_simil, unique_tags_filtered):

        model_ap.set(AffinityPropagation(affinity='precomputed', random_state=123, verbose=True))  #
        model_ap().fit(tag_to_tag_matrix_simil)

        # Get cluster labels
        cluster_labels = model_ap().labels_
        cluster_indices = model_ap().cluster_centers_indices_

        net = Network(height="750px", width="100%", bgcolor="white", directed=True)  # , font_color="white"

        palette = list(np.random.choice(range(256), size=(len(cluster_indices), 3)))

        node_size = []

        for i in range(len(unique_tags_filtered)):
            color_now = palette[cluster_labels[i]]
            color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

            if i in cluster_indices:
                net.add_node(unique_tags_filtered[i], label=unique_tags_filtered[i], color=color_now,
                             size=500)  # ,group=cluster_labels[i]
                node_size.append(1000)
            else:
                net.add_node(unique_tags_filtered[i], label=unique_tags_filtered[i], color=color_now,
                             size=50)  # ,group=cluster_labels[i]
                node_size.append(100)

            net.nodes[i]['font'] = {"size": node_size[i], "color": color_now}

        edges = []

        for i in range(len(unique_tags_filtered)):
            # print(i)
            if i not in cluster_indices:
                temp_edge = (unique_tags_filtered[cluster_indices[cluster_labels[i]]], unique_tags_filtered[i])
                edges.append([temp_edge])

        # Add edges

        for edge in edges:
            net.add_edge(edge[0][0], edge[0][1])

        # Set the physics layout of the network
        net.barnes_hut()

        # Show the network
        net.show_buttons(filter_=['physics'])
        net.write_html("Tag_clustering_vis.html")

        # ui.modal_remove()

        #
        choices = {}
        for i in range(len(cluster_indices)):
            choices[f"clust_choice_{i}"] = f"Cluster {i} (Exemplar: {unique_tags_filtered[cluster_indices[i]]})"
        # cluster_no_sel
        ui.update_selectize(id="cluster_no_sel", choices=choices)


    def spectral_clust_model_vis(tag_to_tag_matrix_simil,unique_tags_filtered,diag_values_tag,no_clusters=10):


        model_ap.set(SpectralClustering(n_clusters=no_clusters, affinity="precomputed", random_state=123).fit(tag_to_tag_matrix_simil))
        max_freq=[]
        model_ap().cluster_centers_indices_=[]
        for i in range(model_ap().n_clusters):
            max_freq.append(0)
            model_ap().cluster_centers_indices_.append(0)
        for i in range(len(model_ap().labels_)):
            temp=model_ap().labels_[i]
            if diag_values_tag[i]>max_freq[temp]:
                model_ap().cluster_centers_indices_[temp]=i
                max_freq[temp]=diag_values_tag[i]


        # Get cluster labels
        cluster_labels = model_ap().labels_
        cluster_indices = model_ap().cluster_centers_indices_

        net = Network(height="750px", width="100%", bgcolor="white", directed=True)  # , font_color="white"

        palette = list(np.random.choice(range(256), size=(len(cluster_indices), 3)))

        node_size = []

        for i in range(len(unique_tags_filtered)):
                    color_now = palette[cluster_labels[i]]
                    color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                    if i in cluster_indices:
                        net.add_node(unique_tags_filtered[i], label=unique_tags_filtered[i], color=color_now,
                                     size=500)  # ,group=cluster_labels[i]
                        node_size.append(1000)
                    else:
                        net.add_node(unique_tags_filtered[i], label=unique_tags_filtered[i], color=color_now,
                                     size=50)  # ,group=cluster_labels[i]
                        node_size.append(100)

                    net.nodes[i]['font'] = {"size": node_size[i], "color": color_now}

        edges = []

        for i in range(len(unique_tags_filtered)):
                    # print(i)
                    if i not in cluster_indices:
                        temp_edge = (unique_tags_filtered[cluster_indices[cluster_labels[i]]], unique_tags_filtered[i])
                        edges.append([temp_edge])

        # Add edges

        for edge in edges:
                net.add_edge(edge[0][0], edge[0][1])

        # Set the physics layout of the network
        net.barnes_hut()

        # Show the network
        net.show_buttons(filter_=['physics'])
        net.write_html("Tag_clustering_vis.html")

        # ui.modal_remove()

        #
        choices = {}
        for i in range(len(cluster_indices)):
                    choices[f"clust_choice_{i}"] = f"Cluster {i} (Exemplar: {unique_tags_filtered[cluster_indices[i]]})"
        # cluster_no_sel
        ui.update_selectize(id="cluster_no_sel", choices=choices)


    def dbscan_clust_model_vis(tag_to_tag_matrix_simil,unique_tags_filtered,diag_values_tag,min_simil=0.1):


        model_ap.set(DBSCAN(min_samples=2,eps=1-min_simil, metric="precomputed").fit(1-tag_to_tag_matrix_simil))
        max_freq=[]
        model_ap().cluster_centers_indices_=[]
        if -1 in model_ap().labels_:
            model_ap().labels_=model_ap().labels_+1

        model_ap().n_clusters=max(model_ap().labels_)+1

        for i in range(model_ap().n_clusters):
            max_freq.append(0)
            model_ap().cluster_centers_indices_.append(0)

        for i in range(len(model_ap().labels_)):
            temp=model_ap().labels_[i]
            if diag_values_tag[i]>max_freq[temp]:
                model_ap().cluster_centers_indices_[temp]=i
                max_freq[temp]=diag_values_tag[i]


        # Get cluster labels
        cluster_labels = model_ap().labels_
        cluster_indices = model_ap().cluster_centers_indices_

        net = Network(height="750px", width="100%", bgcolor="white", directed=True)  # , font_color="white"

        palette = list(np.random.choice(range(256), size=(len(cluster_indices), 3)))

        node_size = []

        for i in range(len(unique_tags_filtered)):
                    color_now = palette[cluster_labels[i]]
                    color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                    if i in cluster_indices:
                        net.add_node(unique_tags_filtered[i], label=unique_tags_filtered[i], color=color_now,
                                     size=500)  # ,group=cluster_labels[i]
                        node_size.append(1000)
                    else:
                        net.add_node(unique_tags_filtered[i], label=unique_tags_filtered[i], color=color_now,
                                     size=50)  # ,group=cluster_labels[i]
                        node_size.append(100)

                    net.nodes[i]['font'] = {"size": node_size[i], "color": color_now}

        edges = []

        for i in range(len(unique_tags_filtered)):
                    # print(i)
                    if i not in cluster_indices:
                        temp_edge = (unique_tags_filtered[cluster_indices[cluster_labels[i]]], unique_tags_filtered[i])
                        edges.append([temp_edge])

        # Add edges

        for edge in edges:
                net.add_edge(edge[0][0], edge[0][1])

        # Set the physics layout of the network
        net.barnes_hut()

        # Show the network
        net.show_buttons(filter_=['physics'])
        net.write_html("Tag_clustering_vis.html")

        # ui.modal_remove()

        #
        choices = {}
        for i in range(len(cluster_indices)):
                    choices[f"clust_choice_{i}"] = f"Cluster {i} (Exemplar: {unique_tags_filtered[cluster_indices[i]]})"
        # cluster_no_sel
        ui.update_selectize(id="cluster_no_sel", choices=choices)


    tag_cluster_pop_metrics_react=reactive.value()
    tag_cluster_dif_metrics_react=reactive.value()
    tag_cluster_growth_react = reactive.value()
    tag_cluster_growth_month_react = reactive.value()

    def principal_component_anal(df,n_components=1):
        # Standardizing the features
        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        # Applying PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        return principal_components

    def pop_dif_fun(main_data,doc_topic_dists,method="all_weight"):

        main_data = main_data.reset_index(drop=True)

        topic_pop_metrics = {"Weight Views": [], "Weight Score": [],  # "Average Favorizations": [],
                             "Weight Comments": []}
        topic_dif_metrics = {"Weight Answers": [], "Weight Answers / Weight Views": [],
                             "Weight with accepted answers": []}


        if method=="all_weight":
            for i in range(doc_topic_dists.shape[1]):
                sum_weights = sum(doc_topic_dists[:, i])

                temp = sum(doc_topic_dists[:, i] * main_data.loc[:, "ViewCount"]) / sum_weights
                topic_pop_metrics["Weight Views"].append(temp)

                temp = sum(doc_topic_dists[:, i] * main_data.loc[:, "Score"]) / sum_weights
                topic_pop_metrics["Weight Score"].append(temp)

                # temp=sum(doc_topic_dists[:,i]*main_data().loc[:,"FavoriteCount"])/sum_weights
                # topic_pop_metrics["Average Favorizations"].append(temp)

                temp = sum(doc_topic_dists[:, i] * main_data.loc[:, "CommentCount"]) / sum_weights
                topic_pop_metrics["Weight Comments"].append(temp)

                temp = sum(doc_topic_dists[:, i] * main_data.loc[:, "AnswerCount"]) / sum_weights
                topic_dif_metrics["Weight Answers"].append(temp)

                temp = topic_dif_metrics["Weight Answers"][i] / topic_pop_metrics["Weight Views"][i]
                topic_dif_metrics["Weight Answers / Weight Views"].append(temp)

                temp = 0
                for j in range(len(main_data)):
                    if not np.isnan(main_data.loc[j, "AcceptedAnswerId"]):
                        temp = temp + doc_topic_dists[j, i]
                temp = temp / sum_weights

                topic_dif_metrics["Weight with accepted answers"].append(temp)
        elif method=="max_weight":
            num_rows, num_columns = doc_topic_dists.shape
            max_columns = np.argmax(doc_topic_dists, axis=1)
            max_columns_len = []
            for i in range(num_columns):
                max_columns_len.append(list(max_columns).count(i))

                topic_pop_metrics["Weight Views"].append(0)
                topic_pop_metrics["Weight Score"].append(0)
                # topic_pop_metrics["Average Favorizations"].append(0)
                topic_pop_metrics["Weight Comments"].append(0)

                topic_dif_metrics["Weight Answers"].append(0)
                topic_dif_metrics["Weight Answers / Weight Views"].append(0)
                topic_dif_metrics["Weight with accepted answers"].append(0)

            for i in range(num_rows):
                topic_pop_metrics["Weight Views"][max_columns[i]] = topic_pop_metrics["Weight Views"][max_columns[i]] + main_data.loc[i, "ViewCount"] / max_columns_len[max_columns[i]]
                topic_pop_metrics["Weight Score"][max_columns[i]] = topic_pop_metrics["Weight Score"][max_columns[i]] + main_data.loc[i, "Score"] / max_columns_len[max_columns[i]]
                # topic_pop_metrics["Average Favorizations"][max_columns[i]]=topic_pop_metrics["Average Favorizations"][max_columns[i]] + main_data().loc[i,"FavoriteCount"]/max_columns_len[max_columns[i]]
                topic_pop_metrics["Weight Comments"][max_columns[i]] = topic_pop_metrics["Weight Comments"][max_columns[i]] + main_data.loc[i, "CommentCount"] / max_columns_len[max_columns[i]]

                topic_dif_metrics["Weight Answers"][max_columns[i]] = topic_dif_metrics["Weight Answers"][max_columns[i]] + main_data.loc[i, "AnswerCount"] / max_columns_len[max_columns[i]]
                topic_dif_metrics["Weight Answers / Weight Views"][max_columns[i]] = topic_dif_metrics["Weight Answers / Weight Views"][max_columns[i]] + (main_data.loc[i, "AnswerCount"] / main_data.loc[i, "ViewCount"]) / max_columns_len[max_columns[i]]

                if not np.isnan(main_data.loc[i, "AcceptedAnswerId"]):
                    topic_dif_metrics["Weight with accepted answers"][max_columns[i]] = topic_dif_metrics["Weight with accepted answers"][max_columns[i]] + 1 / max_columns_len[max_columns[i]]
        elif method=="corr_option":
            for i in range(doc_topic_dists.shape[1]):
                '''
                topic_pop_metrics["Cluster correlation with Views"].append(spearmanr(doc_to_tag_clust_matrix[:,i],np.array(main_data().loc[:,"ViewCount"])))
                topic_pop_metrics["Cluster correlation with Score"].append(spearmanr(doc_to_tag_clust_matrix[:,i],np.array(main_data().loc[:,"Score"])))
                topic_pop_metrics["Cluster correlation with Comments"].append(spearmanr(doc_to_tag_clust_matrix[:,i],np.array(main_data().loc[:,"CommentCount"])))

                topic_dif_metrics["Cluster correlation with Answers"].append(spearmanr(doc_to_tag_clust_matrix[:,i],np.array(main_data().loc[:,"AnswerCount"])))
                topic_dif_metrics["Cluster correlation with Answers / Average Views"].append(spearmanr(doc_to_tag_clust_matrix[:,i],np.array(main_data().loc[:,"AnswerCount"])/np.array(main_data().loc[:,"ViewCount"])))

                '''
                topic_pop_metrics["Weight Views"].append(spearmanr(doc_topic_dists[:,i],np.array(main_data.loc[:,"ViewCount"]))[0])
                topic_pop_metrics["Weight Score"].append(spearmanr(doc_topic_dists[:,i],np.array(main_data.loc[:,"Score"]))[0])
                topic_pop_metrics["Weight Comments"].append(spearmanr(doc_topic_dists[:,i],np.array(main_data.loc[:,"CommentCount"]))[0])

                topic_dif_metrics["Weight Answers"].append(spearmanr(doc_topic_dists[:,i],np.array(main_data.loc[:,"AnswerCount"]))[0])
                topic_dif_metrics["Weight Answers / Weight Views"].append(spearmanr(doc_topic_dists[:,i],np.array(main_data.loc[:,"AnswerCount"])/np.array(main_data.loc[:,"ViewCount"]))[0])


                '''
                 topic_pop_metrics["Cluster correlation with Views"].append(
                    xicor(doc_to_tag_clust_matrix[:, i], np.array(main_data().loc[:, "ViewCount"]),ties=False))
                topic_pop_metrics["Cluster correlation with Score"].append(
                    xicor(doc_to_tag_clust_matrix[:, i], np.array(main_data().loc[:, "Score"]),ties=False))
                topic_pop_metrics["Cluster correlation with Comments"].append(
                    xicor(doc_to_tag_clust_matrix[:, i], np.array(main_data().loc[:, "CommentCount"]),ties=False))

                topic_dif_metrics["Cluster correlation with Answers"].append(
                    xicor(doc_to_tag_clust_matrix[:, i], np.array(main_data().loc[:, "AnswerCount"]),ties=False))
                topic_dif_metrics["Cluster correlation with Answers / Average Views"].append(
                    xicor(doc_to_tag_clust_matrix[:, i],
                              np.array(main_data().loc[:, "AnswerCount"]) / np.array(main_data().loc[:, "ViewCount"]),ties=False))

                '''
                with_accepted_answer = np.ones(len(main_data))
                with_accepted_answer[np.where(np.isnan(main_data.loc[:, 'AcceptedAnswerId']))] = 0
                topic_dif_metrics["Weight with accepted answers"].append(sum(doc_topic_dists[:,i]*with_accepted_answer)/sum(doc_topic_dists[:,i]))

        # Principal components
        topic_pop_metrics["Popularity Principal component"]=list(principal_component_anal(pd.DataFrame(topic_pop_metrics),n_components=1)[:,0])
        topic_dif_metrics["Difficulty Principal component"]=list(principal_component_anal(pd.DataFrame(topic_dif_metrics),n_components=1)[:,0])

        return topic_pop_metrics , topic_dif_metrics

    def calc_growth_fun(main_data,doc_topic_dists,method="all_weight",label="Topic"):

        main_data = main_data.reset_index(drop=True)

        #year_month=np.zeros((len(main_data()),2))
        years_shown=[]
        month_shown=[]

        topic_growth={}
        topic_growth_month={}
        for i in range(doc_topic_dists.shape[1]):
            topic_growth[f"{label} {i}"]={}
            topic_growth_month[f"{label} {i}"]={}

        if method=="all_weight":
            for i in range(len(main_data)):
                temp = main_data.loc[i, 'CreationDate'].split("-")
                temp[1] = temp[0] + "-" + temp[1]
                temp[0] = int(temp[0])
                # year_month[i, 0] = str(temp[0])
                # year_month[i, 1] = temp[1]
                if temp[0] not in years_shown:
                    years_shown.append(temp[0])
                    month_shown.append(temp[1])
                    for j in range(doc_topic_dists.shape[1]):
                        topic_growth[f"{label} {j}"][temp[0]] = doc_topic_dists[i, j]
                        topic_growth_month[f"{label} {j}"][temp[1]] = doc_topic_dists[i, j]
                else:
                    if temp[1] not in month_shown:
                        month_shown.append(temp[1])
                        for j in range(doc_topic_dists.shape[1]):
                            topic_growth[f"{label} {j}"][temp[0]] = topic_growth[f"{label} {j}"][temp[0]] + doc_topic_dists[
                                i, j]
                            topic_growth_month[f"{label} {j}"][temp[1]] = doc_topic_dists[i, j]
                    else:
                        for j in range(doc_topic_dists.shape[1]):
                            topic_growth[f"{label} {j}"][temp[0]] = topic_growth[f"{label} {j}"][temp[0]] + doc_topic_dists[
                                i, j]
                            topic_growth_month[f"{label} {j}"][temp[1]] = topic_growth_month[f"{label} {j}"][temp[1]] + \
                                                                        doc_topic_dists[i, j]

        elif method=="max_weight":
                max_columns = np.argmax(doc_topic_dists, axis=1)
                for i in range(len(main_data)):
                    temp = main_data.loc[i, 'CreationDate'].split("-")
                    temp[1] = temp[0] + "-" + temp[1]
                    temp[0] = int(temp[0])

                    # year_month[i, 0] = str(temp[0])
                    # year_month[i, 1] = temp[1]
                    '''
                    if temp[0] not in years_shown:
                        years_shown.append(temp[0])
                    if temp[1] not in month_shown:
                        month_shown.append(temp[1])
                    '''

                    if temp[0] not in topic_growth[f"{label} {max_columns[i]}"].keys():
                        topic_growth[f"{label} {max_columns[i]}"][temp[0]] = 1
                    else:
                        topic_growth[f"{label} {max_columns[i]}"][temp[0]] = topic_growth[f"{label} {max_columns[i]}"][
                                                                               temp[0]] + 1

                    if temp[1] not in topic_growth_month[f"{label} {max_columns[i]}"].keys():
                        topic_growth_month[f"{label} {max_columns[i]}"][temp[1]] = 1
                    else:
                        topic_growth_month[f"{label} {max_columns[i]}"][temp[1]] = \
                        topic_growth_month[f"{label} {max_columns[i]}"][temp[1]] + 1

                # Create the plot
        return topic_growth , topic_growth_month

    def reg_model_res(main_data,doc_topic_dists, output_opt, model_opt,label="Topic"):
        if output_opt == "views_opt":
            y = main_data.loc[:, "ViewCount"]
        elif output_opt == "score_opt":
            y = main_data.loc[:, "Score"]
        elif output_opt == "comm_opt":
            y = main_data.loc[:, "CommentCount"]
        elif output_opt == "answer_opt":
            y = main_data.loc[:, "AnswerCount"]
        elif output_opt == "ans_view_opt":
            y = main_data.loc[:, "AnswerCount"] / main_data.loc[:, "ViewCount"]
        elif output_opt == "acc_opt":
            y = np.ones(len(main_data))
            y[np.where(np.isnan(main_data.loc[:, 'AcceptedAnswerId']))] = 0
        elif output_opt == "year_opt":
            y = main_data.loc[:, "year"]
        elif output_opt == "time_opt":
            y = main_data.loc[:, "timestamp"]

        df_table = {}
        df_table[label] = []
        for i in range(doc_topic_dists.shape[1]):
            df_table[label].append(f"{label} {i}")

        if model_opt == "rf_opt":

            reg_model = RandomForestRegressor(n_estimators=11, random_state=42)
            reg_model.fit(doc_topic_dists, y)
            df_table["Feature Importance"] = reg_model.feature_importances_
        elif model_opt == "grad_opt":

            reg_model = GradientBoostingRegressor(random_state=42)
            reg_model.fit(doc_topic_dists, y)
            df_table["Feature Importance"] = reg_model.feature_importances_
        elif model_opt == "dec_tree_opt":

            reg_model = DecisionTreeRegressor(random_state=42)
            reg_model.fit(doc_topic_dists, y)
            df_table["Feature Importance"] = reg_model.feature_importances_
        elif model_opt == "pois_opt":
            reg_model = sm.GLM(y, doc_topic_dists, family=sm.families.Poisson()).fit()
            df_table['Coefficient'] = reg_model.params
            df_table['pvalue'] = reg_model.pvalues
        elif model_opt == "bin_opt":
            reg_model = sm.GLM(y, doc_topic_dists, family=sm.families.Binomial()).fit()
            df_table['Coefficient'] = reg_model.params
            df_table['pvalue'] = reg_model.pvalues
        elif model_opt == "neg_bin_opt":
            reg_model = sm.GLM(y, doc_topic_dists, family=sm.families.NegativeBinomial()).fit()
            df_table['Coefficient'] = reg_model.params
            df_table['pvalue'] = reg_model.pvalues
        elif model_opt == "zero_infl_poiss":
            reg_model = sm.ZeroInflatedPoisson(y, doc_topic_dists ).fit()
            df_table['Coefficient'] = reg_model.params[1:]
            df_table['pvalue'] = reg_model.pvalues[1:]
        elif model_opt == "zero_neg_bin_opt":
            reg_model = sm.ZeroInflatedNegativeBinomialP(y, doc_topic_dists ).fit()
            df_table['Coefficient'] = reg_model.params[1:(len(reg_model.params)-1)]
            df_table['pvalue'] = reg_model.pvalues[1:(len(reg_model.params)-1)]
        elif model_opt == "linear_opt":
            reg_model = sm.OLS(y, doc_topic_dists).fit()
            df_table['Coefficient'] = reg_model.params
            df_table['pvalue'] = reg_model.pvalues

        return df_table


    # Define a function to safely convert strings to float (years)
    def extract_year(value):
            try:
                # Convert the first four characters to a float (representing the year)
                return float(value[0:4])
            except (ValueError, TypeError):
                # Return NaN if the conversion fails (e.g., malformed dates)
                return np.nan

    # datetime
    def extract_timestamp(value):
        numeric_date = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
        return numeric_date

    #month
    def extract_month(value):
            try:
                # Convert the first four characters to a float (representing the year)
                return float(value[0:6])
            except (ValueError, TypeError):
                # Return NaN if the conversion fails (e.g., malformed dates)
                return np.nan

    @reactive.effect
    @reactive.event(input.load_file)
    def _():
        main_data.set(pd.read_csv(input.load_file()[0]['datapath']))

        tag_list = main_data()['Tags'].to_list()
        for i in range(len(tag_list)):
            temp = tag_list[i]
            temp = str(temp)
            temp = temp.replace("<", "")
            temp = temp.split(">")
            temp = temp[0: (len(temp) - 1)]
            tag_list[i] = temp

        unique_tags.set(list(set(list(chain(*tag_list)))))

        num_rows = len(main_data())
        num_columns = len(unique_tags())

        document_tag_matrix = np.zeros((num_rows, num_columns))

        for i in range(len(tag_list)):
            temp = tag_list[i]
            for j in temp:
                index_pos = unique_tags().index(j)
                document_tag_matrix[i, index_pos] = 1

        doc_to_tag_matrix.set(document_tag_matrix)
        tag_to_tag_matrix.set(np.dot(np.transpose(document_tag_matrix), document_tag_matrix))



        main_data()["year"] = np.array(main_data()['CreationDate'].copy().apply(extract_year))



        scaler = MinMaxScaler()
        main_data()["timestamp"]=np.array(main_data()['CreationDate'].copy().apply(extract_timestamp))
        main_data()["timestamp"]=scaler.fit_transform(np.array(main_data()["timestamp"].copy()).reshape(-1, 1))


        @render.text
        def no_posts_head():
            return "No posts: "+ str(len(main_data()))
        @render.text
        def no_tags_head():

            return "No tags: "+ str(len(unique_tags()))
        @render.text
        def no_users_head():
            len_user_list = main_data()['OwnerUserId'].to_list()
            len_user_list=len(set(len_user_list))
            return "No creators/owners: "+ str(len_user_list)
        @render.text
        def no_views_head():
            sum_list = sum(main_data()['ViewCount'].to_list())
            return "No Views: "+ str(sum_list)
        @render.text
        def no_answers_head():
            sum_list = sum(main_data()['AnswerCount'].to_list())
            return "No Answers: " + str(sum_list)
        @render.text
        def no_comments_head():
            sum_list = sum(main_data()['CommentCount'].to_list())
            return "No Comments: " + str(sum_list)

    @render_widget
    def month_year_posts():

        years_months=calc_growth_fun(main_data(), np.ones((len(main_data()),1)),
                        method="all_weight", label="ALL")


        if input.month_year_posts_sel() == "year_opt":
            keys_now=list(years_months[0]['ALL 0'].keys())
            values_now=list(years_months[0]['ALL 0'].values())
            view_now="Year"
        elif input.month_year_posts_sel() == "month_opt":
            keys_now=list(years_months[1]['ALL 0'].keys())
            values_now=list(years_months[1]['ALL 0'].values())
            view_now="Month"


        fig = go.Figure()

        fig.add_trace(go.Scatter(
                    x=(keys_now),
                    y=(values_now),
                    mode='lines+markers',
                    name=f"Posts per {view_now}",
                    line=dict(color="black"),
                    marker=dict(color="black", size=8)
                ))

            # Add titles and labels
        fig.update_layout(
                title=f'Questions posted per {view_now}',
                xaxis_title=view_now,
                yaxis_title='Posts'
               #, template='plotly_white'
            )
        return (fig)



    @render_widget
    def barplot_view_count():
        counts_df = Counter(main_data()['ViewCount'].to_list())
        # Convert counts to a DataFrame for Plotly Express
        counts_df = pd.DataFrame(counts_df.items(), columns=['x', 'y'])

        # Create bar plot
        fig = px.histogram(data_frame=counts_df, nbins=input.barplot_bins_views(), x='x', y='y',color_discrete_sequence=['lightblue'],
                           labels={'x': 'No views', 'y': 'Count'})

        return (fig)  # fig.show()

    @render_widget
    def barplot_answer_count():
        counts_df = Counter(main_data()['AnswerCount'].to_list())
        # Convert counts to a DataFrame for Plotly Express
        counts_df = pd.DataFrame(counts_df.items(), columns=['x', 'y'])

        # Create bar plot
        fig = px.histogram(data_frame=counts_df, nbins=input.barplot_bins_answers(), x='x', y='y',color_discrete_sequence=['darkred'],
                           labels={'x': 'No answers', 'y': 'Count'})

        return (fig)  # fig.show()

    @render_widget
    def barplot_comment_count():
        counts_df = Counter(main_data()['CommentCount'].to_list())
        # Convert counts to a DataFrame for Plotly Express
        counts_df = pd.DataFrame(counts_df.items(), columns=['x', 'y'])

        # Create bar plot
        fig = px.histogram(data_frame=counts_df, nbins=input.barplot_bins_comments(), x='x', y='y',color_discrete_sequence=['orange'],
                           labels={'x': 'No comments', 'y': 'Count'})

        return (fig)  # fig.show()

    @render_widget
    def barplot_score_count():
        counts_df = Counter(main_data()['Score'].to_list())
        # Convert counts to a DataFrame for Plotly Express
        counts_df = pd.DataFrame(counts_df.items(), columns=['x', 'y'])

        # Create bar plot
        fig = px.histogram(data_frame=counts_df, nbins=input.barplot_bins_score(), x='x', y='y',color_discrete_sequence=['lightgreen'],
                           labels={'x': 'Score', 'y': 'Count'})

        return (fig)  # fig.show()


    @render.image
    def main_image_head():
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str("images/logo_main.png"), "width": "100px", "height": "100px"}
        return img

    @render.data_frame
    def main_data_table():
        return (render.DataGrid(main_data()))
        #return main_data()

    @render.data_frame
    def tag_frequency_table():
        diag_values_tag = pd.DataFrame(unique_tags(), columns=["Tag"])
        diag_values_tag['Count'] = np.diagonal(tag_to_tag_matrix()[:, :]).copy()

        diag_values_tag = diag_values_tag.sort_values(by="Count", ascending=False)

        return(diag_values_tag)

    @render.plot
    def tag_wordcloud_output(alt="Tag wordcloud"):

        diag_values_tag = pd.DataFrame(unique_tags(), columns=["Tag"])
        diag_values_tag['Count'] = np.diagonal(tag_to_tag_matrix()[:, :]).copy()

        diag_values_tag = diag_values_tag.sort_values(by="Count", ascending=False)

        # Convert DataFrame to dictionary
        diag_values_tag = dict(zip(diag_values_tag['Tag'], diag_values_tag['Count']))

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(diag_values_tag)


        fig=plt.imshow(wordcloud,interpolation="bilinear")

        return(fig)


    @reactive.effect
    @reactive.event(input.txt_preprocessing_button)
    def _():



        if input.text_content_options()=="text_only":
            main_data()["new_text"]=main_data()["Title"]
        elif input.text_content_options()=="body_only":
            main_data()["new_text"] = main_data()["Body"]
        elif input.text_content_options()=="text_body":
            main_data()["new_text"] = main_data()["Title"] + '. ' + main_data()["Body"]

        main_data.set(text_preprocessing_fun(docs_df=main_data(),txt_preprocessing_choices=input.txt_preprocessing_choices()))

        #print(main_data()['new_text'][0])


        @render.plot
        def word_wordcloud_output(alt="Word wordcloud"):

            # Generate word cloud

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(main_data().new_text.str.cat(sep=' '))

            fig = plt.imshow(wordcloud, interpolation="bilinear")

            return (fig)

        @render.data_frame
        def word_frequency_table(alt="Word Frequencies"):
            temp_text = ' '.join(main_data()['new_text'])

            # Tokenize the text into words
            temp_text = word_tokenize(temp_text)

            # Create frequency distribution
            freq_dist = FreqDist(temp_text)

            # Convert frequency distribution to Pandas DataFrame
            freq_dist = pd.DataFrame(list(freq_dist.items()), columns=['Word', 'Count'])

            # Sort DataFrame by frequency in descending order
            freq_dist = freq_dist.sort_values(by='Count', ascending=False)

            return (freq_dist)

    @reactive.effect
    @reactive.event(input.train_topic_model_button)
    def _():
            no_topics = input.no_topics_option()
            choices={}
            for i in range(no_topics):
                choices[i]=f"Topic {i}"


            # initialize corpus
            corp = Corpus()

            # add the processed text in the corpus
            zero_indexes = []
            k = 0
            # add the processed text in the corpus
            for i in main_data()['new_text']:
                k = k + 1
                if len(i) != 0:
                    corp.add_doc(i.split())
                else:
                    zero_indexes.append(k)


            zero_indexes_list.set(zero_indexes)

            min_df = input.min_df_opt()
            rm_top = input.rm_top_opt()
            alpha_prior = 0.05
            eta_prior = 0.001



            if input.topic_model_term_weighting()=="bin_topic_model_term_weighting":
                #print(str(1) + " Binary")
                tw_tm=tp.TermWeight.ONE
            elif input.topic_model_term_weighting()=="idf_topic_model_term_weighting":
                #print(str(2) + " IDF")
                tw_tm = tp.TermWeight.IDF
            elif input.topic_model_term_weighting()=="pmi_topic_model_term_weighting":
                tw_tm = tp.TermWeight.PMI
                #(str(3) + " PMI")




            # Train topic model
            if input.topic_model_option()=="lda_model_option":
                mdl = tp.LDAModel(tw=tw_tm, min_df=min_df, k=no_topics, corpus=corp, seed=123, rm_top=rm_top,alpha=alpha_prior, eta=eta_prior)  # ,rm_top=20
            elif input.topic_model_option()=="ctm_model_option":
                mdl = tp.CTModel(tw=tw_tm, min_df=min_df, k=no_topics, corpus=corp, seed=123, rm_top=rm_top, eta=eta_prior)  # ,rm_top=20
            elif input.topic_model_option()=="ptm_model_option":
                mdl= tp.PTModel(tw=tw_tm, min_df=min_df, k=no_topics, corpus=corp, seed=123, rm_top=rm_top,alpha=alpha_prior, eta=eta_prior)
            elif input.topic_model_option()=="dmr_model_option":
                mdl= tp.DMRModel(tw=tw_tm, min_df=min_df, k=no_topics, corpus=corp, seed=123, rm_top=rm_top,alpha=alpha_prior, eta=eta_prior)




            mdl.train(input.no_iterations_topic_model())




            @reactive.effect
            @reactive.event(input.no_topic_top_words)
            def _():
                # Model coherence values go from 0-1 when c_v is selected - the higher the better
                topic_coherence_now = (tp.coherence.Coherence(mdl, coherence="c_v", top_n=input.no_topic_top_words()).get_score())

                temp = []
                for j in range(0, (no_topics)):
                    temp_2 = mdl.get_topic_words(j,top_n=input.no_topic_top_words())
                    for k in temp_2:
                        temp.append(k[0])
                temp = np.unique(temp)

                topic_divergence_now = (len(temp) / (input.no_topic_top_words() * no_topics))


                @render.text
                def topic_coherence_output_text():
                    return f"Topic coherence C_V for {no_topics} topics and {input.no_topic_top_words()} top words: {topic_coherence_now}"

                @render.text
                def topic_divergence_output_text():
                    return f"Topic Divergence between top words for {no_topics} topics and {input.no_topic_top_words()} top words: {topic_divergence_now}"

                @render.text
                def topic_cumscore_output_text():
                    return f"Sum of two scores  for {no_topics} topics and {input.no_topic_top_words()} top words: {(topic_divergence_now + topic_coherence_now)}"


            topic_model.set(mdl)

            choices={}
            for i in range(topic_model().k):
                choices[f"topic_opt_{i}"]=f"Topic {i}"
            ui.update_selectize(id="topdocs_topic_opt",choices=choices)
            #
            @render.ui
            def lda_vis_topic_model():


                #####Variables
                vocab = list(topic_model().used_vocabs)
                topic_term_dists = np.stack([topic_model().get_topic_word_dist(k) for k in range(topic_model().k)])
                doc_topic_dists = np.stack([doc.get_topic_dist() for doc in topic_model().docs])
                doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
                doc_lengths = np.array([len(doc.topics) for doc in topic_model().docs])
                term_frequency = topic_model().used_vocab_weighted_freq

                #

                # top_words_weights=np.zeros([10,mdl.k],dtype=str)
                top_words_weights = pd.DataFrame(index=range(30), columns=range(topic_model().k), dtype=str)
                for i in range(topic_model().k):
                    temp = topic_model().get_topic_words(i, 30)
                    temp_2 = []
                    for j in range(len(temp)):
                        temp_2 = temp[j]
                        top_words_weights.loc[j, i] = str(temp_2[0] + " " + str(temp_2[1]))

                top_words_weights.columns = ["Topic_" + str(i) for i in range(topic_model().k)]

                prepare_data_topic_model_ldavis=pyLDAvis.prepare(
                    topic_term_dists=topic_term_dists,
                    doc_topic_dists=doc_topic_dists,
                    doc_lengths=doc_lengths,
                    vocab=vocab,
                    term_frequency=term_frequency,
                    start_index=0,  # tomotopy starts topic ids with 0, pyLDAvis with 1
                    sort_topics=False  # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
                )
                prepare_data_topic_model_ldavis= pyLDAvis.prepared_data_to_html( prepare_data_topic_model_ldavis)

                return ui.TagList(
                    ui.HTML(prepare_data_topic_model_ldavis)
                )

    #topdocs_topic_vis_table
    @reactive.effect
    @reactive.event(input.topdocs_topic_button)
    def _():

        #Selected topic
        sel_topic=int(input.topdocs_topic_opt().split("_")[2])

        # Calculate weights
        doc_weights = np.stack([doc.get_topic_dist() for doc in topic_model().docs]) #normalize=False

        doc_weights=doc_weights[:,sel_topic]

        # Step 2: Get the indices of the top 10 values
        top_10_indices = np.argpartition(doc_weights, -10)[-10:]

        # Step 3: Sort these indices to get them in the correct order
        top_10_indices = top_10_indices[np.argsort(doc_weights[top_10_indices])[::-1]]

        #Only top10 weights
        doc_weights=doc_weights[top_10_indices]
        @render.table
        def topdocs_topic_vis_table():
            df={}
            df["Index"]=top_10_indices
            df["Score"]=doc_weights
            df["Title"]=main_data().loc[top_10_indices,"Title"]
            df["Processed Text"] = main_data().loc[top_10_indices, "new_text"]

            return(pd.DataFrame(df))

    @reactive.effect
    @reactive.event(input.pop_dif_metrics_button_topic)
    def _():
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in topic_model().docs])
        #doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)

        if input.pop_dif_weight_option()=="max_weight":
            topic_pop_metrics,  topic_dif_metrics= pop_dif_fun(main_data().loc[~main_data().index.isin(zero_indexes_list()),:],doc_topic_dists,method="max_weight")
        elif input.pop_dif_weight_option()=="all_weight":
            topic_pop_metrics,  topic_dif_metrics= pop_dif_fun(main_data().loc[~main_data().index.isin(zero_indexes_list()),:],doc_topic_dists,method="all_weight")
        elif input.pop_dif_weight_option()=="corr_option":
            topic_pop_metrics,topic_dif_metrics=  pop_dif_fun(main_data().loc[~main_data().index.isin(zero_indexes_list()),:],doc_topic_dists,method="corr_option")

        topic_pop_metrics_react.set(topic_pop_metrics)
        topic_dif_metrics_react.set(topic_dif_metrics)

        #@render.data_frame

        @render.table
        def pop_metrics_table_topic():
            #return render.DataGrid(pd.DataFrame(topic_pop_metrics) )#, selection_mode="rows"
            return pd.DataFrame(topic_pop_metrics)

        @render.table
        def dif_metrics_table_topic():
            return pd.DataFrame(topic_dif_metrics)


    @reactive.effect
    @reactive.event(input.topic2d_metric_button)
    def _():
        # topic_2dmetrics_1
        # ,choices={"views_opt":"Average Views","score_opt":"Average Score","comm_opt":"Average Comments","answer_opt":"Average Answers","answer_views_opt":"Average Answers / Views","acc_opt":"Weight with accepted answers"}))
        # topic_pop_metrics = {"Average Views": [], "Average Score": [], #"Average Favorizations": [],"Average Comments": []}
        # topic_dif_metrics = {"Average Answers": [], "Average Answers / Average Views": [],"Weight with accepted answers": []}
        df = {}
        if input.topic_2dmetrics_1() == "views_opt":
            df["Weight Views"] = topic_pop_metrics_react()["Weight Views"]
            x_val = "Weight Views"
        elif input.topic_2dmetrics_1() == "score_opt":
            df["Weight Score"] = topic_pop_metrics_react()["Weight Score"]
            x_val = "Weight Score"
        elif input.topic_2dmetrics_1() == "comm_opt":
            df["Weight Comments"] = topic_pop_metrics_react()["Weight Comments"]
            x_val = "Weight Comments"
        elif input.topic_2dmetrics_1() == "answer_opt":
            df["Weight Answers"] = topic_dif_metrics_react()["Weight Answers"]
            x_val = "Weight Answers"
        elif input.topic_2dmetrics_1() == "answer_views_opt":
            df["Weight Answers / Weight Views"] = topic_dif_metrics_react()["Weight Answers / Weight Views"]
            x_val = "Weight Answers / Weight Views"
        elif input.topic_2dmetrics_1() == "acc_opt":
            df["Weight with accepted answers"] = topic_dif_metrics_react()["Weight with accepted answers"]
            x_val = "Weight with accepted answers"
        elif input.topic_2dmetrics_1() == "poppc_opt":
            df["Popularity Principal component"] = topic_pop_metrics_react()["Popularity Principal component"]
            x_val = "Popularity Principal component"
        elif input.topic_2dmetrics_1() == "difpc_opt":
            df["Difficulty Principal component"] = topic_dif_metrics_react()["Difficulty Principal component"]
            x_val = "Difficulty Principal component"

        if input.topic_2dmetrics_2() == "views_opt":
            df["Weight Views"] = topic_pop_metrics_react()["Weight Views"]
            y_val = "Weight Views"
        elif input.topic_2dmetrics_2() == "score_opt":
            df["Weight Score"] = topic_pop_metrics_react()["Weight Score"]
            y_val = "Weight Score"
        elif input.topic_2dmetrics_2() == "comm_opt":
            df["Weight Comments"] = topic_pop_metrics_react()["Weight Comments"]
            y_val = "Weight Comments"
        elif input.topic_2dmetrics_2() == "answer_opt":
            df["Weight Answers"] = topic_dif_metrics_react()["Weight Answers"]
            y_val = "Weight Answers"
        elif input.topic_2dmetrics_2() == "answer_views_opt":
            df["Weight Answers / Weight Views"] = topic_dif_metrics_react()["Weight Answers / Weight Views"]
            y_val = "Weight Answers / Weight Views"
        elif input.topic_2dmetrics_2() == "acc_opt":
            df["Weight with accepted answers"] = topic_dif_metrics_react()["Weight with accepted answers"]
            y_val = "Weight with accepted answers"
        elif input.topic_2dmetrics_2() == "poppc_opt":
            df["Popularity Principal component"] = topic_pop_metrics_react()["Popularity Principal component"]
            y_val = "Popularity Principal component"
        elif input.topic_2dmetrics_2() == "difpc_opt":
            df["Difficulty Principal component"] = topic_dif_metrics_react()["Difficulty Principal component"]
            y_val = "Difficulty Principal component"

        text_list = []
        for i in range(topic_model().k):
            text_list.append(f"Topic {i}")

        fig = px.scatter(df, x=x_val, y=y_val,
                         text=text_list)  # , color="species" , size='petal_length' , hover_data=['petal_width']
        @render_widget
        def topic2d_vis_widget():


            return(fig)


    @reactive.effect
    @reactive.event(input.growth_metrics_button_topic)
    def _():
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in topic_model().docs])


        if input.topic_growth_weight_option()=="all_weight":
            topic_growth , topic_growth_month = calc_growth_fun(main_data().loc[~main_data().index.isin(zero_indexes_list()),:],doc_topic_dists,method="all_weight",label="Topic")
        elif input.topic_growth_weight_option()=="max_weight":
            topic_growth , topic_growth_month = calc_growth_fun(main_data().loc[~main_data().index.isin(zero_indexes_list()),:],doc_topic_dists,method="max_weight",label="Topic")


        topic_growth_react.set(topic_growth)
        topic_growth_month_react.set(topic_growth_month)

    @render_widget
    def topic_growth_year_vis():
            # Create the plot
            #print(selected_topic_now)
            if input.topic_model_growth_my()=="per_year":
                fig = go.Figure()
                palette = list(np.random.choice(range(256), size=((topic_model().k), 3)))

                for selected_topic_now in range((topic_model().k)):
                    color_now = palette[selected_topic_now]
                    color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                    topic_growth_react()[f"Topic {selected_topic_now}"] = OrderedDict(sorted(topic_growth_react()[f"Topic {selected_topic_now}"].items()))
                    fig.add_trace(go.Scatter(
                        x=list(topic_growth_react()[f"Topic {selected_topic_now}"].keys()), y=list(topic_growth_react()[f"Topic {selected_topic_now}"].values()),
                        mode='lines+markers',
                        name=f"Topic {selected_topic_now}",
                        line=dict(color=color_now),
                        marker=dict(color=color_now, size=8)
                    ))

                # Add titles and labels
                fig.update_layout(
                    title='Topic Weights Over Years',
                    xaxis_title='Year',
                    yaxis_title='Weight',
                    template='plotly_white'
                )
                return(fig)
            elif input.topic_model_growth_my()=="per_month":
                fig = go.Figure()
                palette = list(np.random.choice(range(256), size=((topic_model().k), 3)))

                for selected_topic_now in range((topic_model().k)):
                    color_now = palette[selected_topic_now]
                    color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                    topic_growth_month_react()[f"Topic {selected_topic_now}"] = OrderedDict(sorted(topic_growth_month_react()[f"Topic {selected_topic_now}"].items()))
                    fig.add_trace(go.Scatter(
                        x=list(topic_growth_month_react()[f"Topic {selected_topic_now}"].keys()),
                        y=list(topic_growth_month_react()[f"Topic {selected_topic_now}"].values()),
                        mode='lines+markers',
                        name=f"Topic {selected_topic_now}",
                        line=dict(color=color_now),
                        marker=dict(color=color_now, size=8)
                    ))

                # Add titles and labels
                fig.update_layout(
                    title='Topic Weights Over Months',
                    xaxis_title='Month',
                    yaxis_title='Weight',
                    template='plotly_white'
                )
                return (fig)

    @reactive.effect
    @reactive.event(input.topic_reg_button)
    def _():
        if input.topic_reg_weight()=="prop_opt":
            doc_topic_dists = np.stack([doc.get_topic_dist() for doc in topic_model().docs])
        elif input.topic_reg_weight()=="weight_opt":
            doc_topic_dists = np.stack([doc.get_topic_dist() * len(doc.topics) for doc in topic_model().docs])

        #{"views_opt": "Views", "score_opt": "Score", "comm_opt":"Comments",
        #"answer_opt": "Answers", "ans_view_opt":"Answers / Views","acc_opt":"Has accepted answer"}


        df_table=reg_model_res(main_data().loc[~main_data().index.isin(zero_indexes_list()),:],doc_topic_dists,output_opt=input.topic_reg_output(),model_opt=input.topic_reg_opt(),label="Topic")



        @render.table
        def topic_reg_table():
            return(pd.DataFrame(df_table))

    @reactive.effect
    @reactive.event(input.tag_clust_options_alg)
    def _():
        if input.tag_clust_options_alg()=="spectral_clust_option":
            ui.update_numeric(id="extra_opt_clust",label="Number of Clusters",value=10,min=2,max=200)
        elif input.tag_clust_options_alg()=="dbscan_clust_option":
            ui.update_numeric(id="extra_opt_clust",label="Minimum similarity",value=0.1,min=0,max=1)




    @reactive.effect
    @reactive.event(input.tag_clust_button)
    def _():
        #m = ui.modal("Tag clustering in progress, please wait....",easy_close=False,size="m",footer=None,)#title="Somewhat important message",
        #ui.modal_show(m)
            thres_min=len(main_data())/100*input.low_thres_tag_clust()
            thres_max=len(main_data())/100*input.up_thres_tag_clust()

            exclude_manual=input.exclude_tag_text().strip()
            exclude_manual=re.sub(r'\s+', ' ',exclude_manual)
            exclude_manual=exclude_manual.split()

            diag_values_tag = np.diagonal(tag_to_tag_matrix()[:, :]).copy()


            include_tags=[]
            for i in range(len(tag_to_tag_matrix())):
                temp_diag=diag_values_tag[i]
                temp_name=unique_tags()[i]
                if temp_name not in exclude_manual:
                    if temp_diag>=thres_min and temp_diag<=thres_max:
                        include_tags.append(i)


            if len(include_tags)>2:
                #print("No tags "+str(len(include_tags)))
                tag_to_tag_matrix_simil = tag_to_tag_matrix().copy()
                tag_to_tag_matrix_simil=tag_to_tag_matrix_simil[np.ix_(include_tags, include_tags)]
                tag_to_tag_matrix_simil = tag_to_tag_matrix_simil.astype(float)

                diag_values_tag=[diag_values_tag[i] for i in include_tags]

                unique_tags_filtered=[unique_tags()[i] for i in include_tags]

                if input.tag_clust_options_weight()=="ii_weight_option":
                    for i in range(len(tag_to_tag_matrix_simil) - 1):
                        for j in range(i + 1, len(tag_to_tag_matrix_simil)):
                            temp_val = min(diag_values_tag[i], diag_values_tag[j])
                            temp_val = tag_to_tag_matrix_simil[i, j] / temp_val
                            tag_to_tag_matrix_simil[i, j] = temp_val
                            tag_to_tag_matrix_simil[j, i] = temp_val
                elif input.tag_clust_options_weight()=="ri_weight_option":
                    for i in range(len(tag_to_tag_matrix_simil) - 1):
                        for j in range(i + 1, len(tag_to_tag_matrix_simil)):
                            temp_val = max(diag_values_tag[i], diag_values_tag[j])
                            temp_val = tag_to_tag_matrix_simil[i, j] / temp_val
                            tag_to_tag_matrix_simil[i, j] = temp_val
                            tag_to_tag_matrix_simil[j, i] = temp_val
                elif input.tag_clust_options_weight()=="ji_weight_option":
                    for i in range(len(tag_to_tag_matrix_simil) - 1):
                        for j in range(i + 1, len(tag_to_tag_matrix_simil)):
                            temp_val = tag_to_tag_matrix_simil[i,j]/(diag_values_tag[i] + diag_values_tag[j] - tag_to_tag_matrix_simil[i,j])

                            tag_to_tag_matrix_simil[i, j] = temp_val
                            tag_to_tag_matrix_simil[j, i] = temp_val
                elif input.tag_clust_options_weight()=="ei_weight_option":
                    for i in range(len(tag_to_tag_matrix_simil) - 1):
                        for j in range(i + 1, len(tag_to_tag_matrix_simil)):
                            temp_val = (tag_to_tag_matrix_simil[i,j]*tag_to_tag_matrix_simil[i,j])/(diag_values_tag[i] * diag_values_tag[j])

                            tag_to_tag_matrix_simil[i, j] = temp_val
                            tag_to_tag_matrix_simil[j, i] = temp_val

                np.fill_diagonal(tag_to_tag_matrix_simil, 1)

                tag_to_tag_matrix_simil = pd.DataFrame(tag_to_tag_matrix_simil)
                tag_to_tag_matrix_simil.columns = unique_tags_filtered


                # Create and fit the AffinityPropagation model
                if input.tag_clust_options_alg()=="affinity_clust_option":
                    affinity_prop_model_vis(tag_to_tag_matrix_simil,unique_tags_filtered)

                elif input.tag_clust_options_alg()=="spectral_clust_option":
                    spectral_clust_model_vis(tag_to_tag_matrix_simil,unique_tags_filtered,diag_values_tag,no_clusters=input.extra_opt_clust())

                elif input.tag_clust_options_alg() == "dbscan_clust_option":
                    dbscan_clust_model_vis(tag_to_tag_matrix_simil, unique_tags_filtered, diag_values_tag,min_simil=input.extra_opt_clust())

    @reactive.effect
    @reactive.event(input.cluster_no_sel)
    def _():
        @render.data_frame
        def cluster_top_tags():

            clust_sel_no=int(input.cluster_no_sel().split("_")[2])



            diag_val=np.diagonal(tag_to_tag_matrix()).copy()
            data_table={"Tag":[],"No documents":[],"No co-occurences with the exemplar":[]}

            temp_pos_ex =model_ap().cluster_centers_indices_[clust_sel_no]
            temp_pos_ex=model_ap().feature_names_in_[temp_pos_ex]
            temp_pos_ex=unique_tags().index(temp_pos_ex)


            for i in range(len(model_ap().labels_)):
                if model_ap().labels_[i]==clust_sel_no:
                    data_table["Tag"].append(model_ap().feature_names_in_[i])

                    temp_pos=unique_tags().index(model_ap().feature_names_in_[i])
                    data_table["No documents"].append(diag_val[temp_pos])

                    data_table["No co-occurences with the exemplar"].append(tag_to_tag_matrix()[temp_pos,temp_pos_ex])

            return(pd.DataFrame(data_table))



    # show_tag_clust_button
    @reactive.effect
    @reactive.event(input.show_tag_clust_button)
    def _():
        with open("Tag_clustering_vis.html", 'r', encoding='utf-8') as file:  #
            html_string = file.read()
        @render.ui
        def tag_clust_vis():

            return (ui.HTML(html_string))

    # show_tag_clust_button
    @reactive.effect
    @reactive.event(input.pop_dif_metrics_button_tag_clust)
    def _():

        #labels_
        #cluster_centers_indices_
        #doc_to_tag_matrix
        #feature_names_in_
        doc_to_tag_clust_matrix=np.zeros(((len(main_data()), len(model_ap().cluster_centers_indices_))))

        for i in range(len(model_ap().feature_names_in_)):
            temp_pos=unique_tags().index(model_ap().feature_names_in_[i])
            doc_to_tag_clust_matrix[:,model_ap().labels_[i]]=doc_to_tag_clust_matrix[:,model_ap().labels_[i]]+doc_to_tag_matrix()[:,temp_pos]

        if input.pop_dif_weight_option_clust()=="corr_option":
            topic_pop_metrics ,   topic_dif_metrics =  pop_dif_fun(main_data(),doc_to_tag_clust_matrix,method="corr_option")
        elif input.pop_dif_weight_option_clust()=="all_weight":
            topic_pop_metrics ,   topic_dif_metrics = pop_dif_fun(main_data(),doc_to_tag_clust_matrix,method="all_weight")
        elif input.pop_dif_weight_option_clust()=="max_weight":
            topic_pop_metrics ,   topic_dif_metrics = pop_dif_fun(main_data(),doc_to_tag_clust_matrix,method="max_weight")


        tag_cluster_pop_metrics_react.set(topic_pop_metrics)
        tag_cluster_dif_metrics_react.set(topic_dif_metrics)



        @render.table
        def pop_metrics_table_tag_clust():

            return (pd.DataFrame(topic_pop_metrics))

        @render.table
        def dif_metrics_table_tag_clust():

            return (pd.DataFrame(topic_dif_metrics))

    @reactive.effect
    @reactive.event(input.tag_clust2d_metric_button)
    def _():
        # topic_2dmetrics_1
        # ,choices={"views_opt":"Average Views","score_opt":"Average Score","comm_opt":"Average Comments","answer_opt":"Average Answers","answer_views_opt":"Average Answers / Views","acc_opt":"Weight with accepted answers"}))
        # topic_pop_metrics = {"Average Views": [], "Average Score": [], #"Average Favorizations": [],"Average Comments": []}
        # topic_dif_metrics = {"Average Answers": [], "Average Answers / Average Views": [],"Weight with accepted answers": []}
        df = {}
        if input.tag_clust_2dmetrics_1() == "views_opt":
            df["Weight Views"] = tag_cluster_pop_metrics_react()["Weight Views"]
            x_val = "Weight Views"
        elif input.tag_clust_2dmetrics_1() == "score_opt":
            df["Weight Score"] = tag_cluster_pop_metrics_react()["Weight Score"]
            x_val = "Weight Score"
        elif input.tag_clust_2dmetrics_1() == "comm_opt":
            df["Weight Comments"] = tag_cluster_pop_metrics_react()["Weight Comments"]
            x_val = "Weight Comments"
        elif input.tag_clust_2dmetrics_1() == "answer_opt":
            df["Weight Answers"] = tag_cluster_dif_metrics_react()["Weight Answers"]
            x_val = "Weight Answers"
        elif input.tag_clust_2dmetrics_1() == "answer_views_opt":
            df["Weight Answers / Weight Views"] = tag_cluster_dif_metrics_react()["Weight Answers / Weight Views"]
            x_val = "Weight Answers / Weight Views"
        elif input.tag_clust_2dmetrics_1() == "acc_opt":
            df["Weight with accepted answers"] = tag_cluster_dif_metrics_react()["Weight with accepted answers"]
            x_val = "Weight with accepted answers"
        elif input.tag_clust_2dmetrics_1() == "poppc_opt":
            df["Popularity Principal component"] = tag_cluster_pop_metrics_react()["Popularity Principal component"]
            x_val = "Popularity Principal component"
        elif input.tag_clust_2dmetrics_1() == "difpc_opt":
            df["Difficulty Principal component"] = tag_cluster_dif_metrics_react()["Difficulty Principal component"]
            x_val = "Difficulty Principal component"

        if input.tag_clust_2dmetrics_2() == "views_opt":
            df["Weight Views"] = tag_cluster_pop_metrics_react()["Weight Views"]
            y_val = "Weight Views"
        elif input.tag_clust_2dmetrics_2() == "score_opt":
            df["Weight Score"] = tag_cluster_pop_metrics_react()["Weight Score"]
            y_val = "Weight Score"
        elif input.tag_clust_2dmetrics_2() == "comm_opt":
            df["Weight Comments"] = tag_cluster_pop_metrics_react()["Weight Comments"]
            y_val = "Weight Comments"
        elif input.tag_clust_2dmetrics_2() == "answer_opt":
            df["Weight Answers"] = tag_cluster_dif_metrics_react()["Weight Answers"]
            y_val = "Weight Answers"
        elif input.tag_clust_2dmetrics_2() == "answer_views_opt":
            df["Weight Answers / Weight Views"] = tag_cluster_dif_metrics_react()["Weight Answers / Average Views"]
            y_val = "Weight Answers / Weight Views"
        elif input.tag_clust_2dmetrics_2() == "acc_opt":
            df["Weight with accepted answers"] = tag_cluster_dif_metrics_react()["Weight with accepted answers"]
            y_val = "Weight with accepted answers"
        elif input.tag_clust_2dmetrics_2() == "poppc_opt":
            df["Popularity Principal component"] = tag_cluster_pop_metrics_react()["Popularity Principal component"]
            y_val = "Popularity Principal component"
        elif input.tag_clust_2dmetrics_2() == "difpc_opt":
            df["Difficulty Principal component"] = tag_cluster_dif_metrics_react()["Difficulty Principal component"]
            y_val = "Difficulty Principal component"

        text_list = []
        for i in range(len(model_ap().cluster_centers_indices_)):
            text_list.append(f"Cluster {i}")

        fig = px.scatter(df, x=x_val, y=y_val,
                         text=text_list)  # , color="species" , size='petal_length' , hover_data=['petal_width']
        @render_widget
        def tag_clust_vis_widget():
            return(fig)

    @reactive.effect
    @reactive.event(input.growth_metrics_button_cluster)
    def _():
        doc_to_tag_clust_matrix = np.zeros(((len(main_data()), len(model_ap().cluster_centers_indices_))))

        for i in range(len(model_ap().feature_names_in_)):
            temp_pos = unique_tags().index(model_ap().feature_names_in_[i])
            doc_to_tag_clust_matrix[:, model_ap().labels_[i]] = doc_to_tag_clust_matrix[:,
                                                                model_ap().labels_[i]] + doc_to_tag_matrix()[:,
                                                                                    temp_pos]

        if input.cluster_growth_weight_option() == "all_weight":
            cluster_growth , cluster_growth_month = calc_growth_fun(main_data(),doc_to_tag_clust_matrix, method="all_weight",label="Cluster")
        elif input.cluster_growth_weight_option() == "max_weight":
            cluster_growth , cluster_growth_month = calc_growth_fun(main_data(),doc_to_tag_clust_matrix, method="max_weight",label="Cluster")


        tag_cluster_growth_react.set(cluster_growth)
        tag_cluster_growth_month_react.set(cluster_growth_month)

    @render_widget
    def cluster_growth_year_vis():
            # Create the plot
            #print(selected_topic_now)

            if input.cluster_model_growth_my()=="per_year":

                fig = go.Figure()
                palette = list(np.random.choice(range(256), size=(len(tag_cluster_growth_react().keys()), 3)))

                for clust_sel_no in range(len(tag_cluster_growth_react().keys())):
                    color_now = palette[clust_sel_no]
                    color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"
                    tag_cluster_growth_react()[f"Cluster {clust_sel_no}"] = OrderedDict(sorted(tag_cluster_growth_react()[f"Cluster {clust_sel_no}"].items()))
                    fig.add_trace(go.Scatter(
                        x=list(tag_cluster_growth_react()[f"Cluster {clust_sel_no}"].keys()), y=list(tag_cluster_growth_react()[f"Cluster {clust_sel_no}"].values()),
                        mode='lines+markers',
                        name= f"Cluster {clust_sel_no}",
                        line=dict(color=color_now), # 'blue'
                        marker=dict(color=color_now, size=8)
                    ))

                # Add titles and labels
                fig.update_layout(
                    title='Cluste Weight Over Years',
                    xaxis_title='Year',
                    yaxis_title='Weight',
                    template='plotly_white'
                )
                return(fig)
            elif input.cluster_model_growth_my()=="per_month":

                fig = go.Figure()
                palette = list(np.random.choice(range(256), size=(len(tag_cluster_growth_month_react().keys()), 3)))

                for clust_sel_no in range(len(tag_cluster_growth_month_react().keys())):
                    color_now = palette[clust_sel_no]
                    color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                    tag_cluster_growth_month_react()[f"Cluster {clust_sel_no}"] = OrderedDict(sorted(tag_cluster_growth_month_react()[f"Cluster {clust_sel_no}"].items()))
                    fig.add_trace(go.Scatter(
                        x=list(tag_cluster_growth_month_react()[f"Cluster {clust_sel_no}"].keys()),
                        y=list(tag_cluster_growth_month_react()[f"Cluster {clust_sel_no}"].values()),
                        mode='lines+markers',
                        name=f"Cluster {clust_sel_no}",
                        line=dict(color=color_now),
                        marker=dict(color=color_now, size=8)
                    ))

                # Add titles and labels
                fig.update_layout(
                    title='Cluster Weight Over Months',
                    xaxis_title='Month',
                    yaxis_title='Weight',
                    template='plotly_white'
                )
                return (fig)

    @reactive.effect
    @reactive.event(input.clust_reg_button)
    def _():

        doc_to_tag_clust_matrix=np.zeros(((len(main_data()), len(model_ap().cluster_centers_indices_))))

        for i in range(len(model_ap().feature_names_in_)):
            temp_pos=unique_tags().index(model_ap().feature_names_in_[i])
            doc_to_tag_clust_matrix[:,model_ap().labels_[i]]=doc_to_tag_clust_matrix[:,model_ap().labels_[i]]+doc_to_tag_matrix()[:,temp_pos]

        df_table=reg_model_res(main_data(),doc_to_tag_clust_matrix,output_opt=input.clust_reg_output(),model_opt=input.clust_reg_opt(),label="Cluster")



        @render.table
        def clust_reg_table():
            return(pd.DataFrame(df_table))


    @reactive.effect
    @reactive.event(input.train_doc_cluster_button)
    def _():

        no_clust_doc.set(input.no_features_option())
        #model_doc
        def create_V_bow(main_data, min_thres=0.001, max_thres=1.0):

            if input.doc_clust_is_bin()=="bin_option":
                bin_option=True
            else:
                bin_option=False

            min_perc=min_thres/100.0
            max_perc=max_thres/100.0
            if input.doc_cluster_feature_options()=="text_tags":

                tag_list = main_data['Tags'].to_list()
                for i in range(len(tag_list)):
                    temp = tag_list[i]
                    temp = temp.replace("-", "_")
                    temp = temp.replace(".", "_")
                    temp = temp.replace("<", "")
                    temp = temp.split(">")
                    temp = temp[0: (len(temp) - 1)]
                    for j in range(len(temp)):
                        temp[j] = "tag_" + temp[j]
                    temp = " ".join(temp)
                    tag_list[i] = temp

                # Create an instance of CountVectorizer with binary=True

                vectorizer = CountVectorizer(binary=bin_option, min_df=min_perc, max_df=max_perc)
                tag_mat = vectorizer.fit_transform(tag_list).toarray()
                unique_tags_filtered = list(vectorizer.get_feature_names_out())


                vectorizer = CountVectorizer(binary=bin_option, min_df=min_perc, max_df=max_perc)
                word_mat = vectorizer.fit_transform(main_data['new_text']).toarray()
                unique_words_filtered = list(vectorizer.get_feature_names_out())

                V = np.column_stack((tag_mat, word_mat)).copy()
                unique_features = unique_tags_filtered + unique_words_filtered
            elif input.doc_cluster_feature_options()=="tags_only":

                tag_list = main_data['Tags'].to_list()
                for i in range(len(tag_list)):
                    temp = tag_list[i]
                    temp = str(temp)
                    temp = temp.replace("-", "_")
                    temp = temp.replace(".", "_")
                    temp = temp.replace("<", "")
                    temp = temp.split(">")
                    temp = temp[0: (len(temp) - 1)]
                    for j in range(len(temp)):
                        temp[j] = "tag_" + temp[j]
                    temp = " ".join(temp)
                    tag_list[i] = temp

                # Create an instance of CountVectorizer with binary=True

                vectorizer = CountVectorizer(binary=bin_option, min_df=min_perc, max_df=max_perc)
                V = vectorizer.fit_transform(tag_list).toarray()
                unique_features = list(vectorizer.get_feature_names_out())

            elif input.doc_cluster_feature_options()=="text_only":

                vectorizer = CountVectorizer(binary=bin_option, min_df=min_perc, max_df=max_perc)
                V = vectorizer.fit_transform(main_data['new_text']).toarray()
                unique_features = list(vectorizer.get_feature_names_out())

            return V, unique_features


        V , unique__features = create_V_bow(main_data(),min_thres=input.low_thres_doc_clust(), max_thres=input.up_thres_doc_clust())


        rank=input.no_features_option()
        iters=input.no_iterations_doc_cluster()
        # Create the BMF model with explicit random initialization
        np.random.seed(12345)
        init_W = np.random.rand(V.shape[0], rank)
        np.random.seed(12345)
        init_H = np.random.rand(rank, V.shape[1])

        np.random.seed(12345)  # Reset the seed before each BMF initialization

        if input.doc_cluster_choices()=="nmf_option":
            mdl=nimfa.Nmf(V, rank=rank, max_iter=iters, seed=None, W=init_W, H=init_H)
        elif input.doc_cluster_choices()=="bmf_option":
            mdl=nimfa.Bmf(V, rank=rank, max_iter=iters, seed=None, W=init_W, H=init_H)
        elif input.doc_cluster_choices()=="pmf_option":
            mdl=nimfa.Pmf(V, rank=rank, max_iter=iters, seed=None, W=init_W, H=init_H)
        elif input.doc_cluster_choices()=="psmf_option":
            mdl=nimfa.Psmf(V, rank=rank, max_iter=iters, seed=None)#, W=init_W, H=init_H

        mdl_fit=mdl()
        model_doc.set( mdl_fit)
        feat_doc.set(unique__features)

        #
        choices = {}
        for i in range((rank)):
            choices[f"feature_choice_{i}"]=f"Feature {i}"
        # cluster_no_sel
        ui.update_selectize(id="doc_cluster_no_sel", choices=choices)

    @reactive.effect
    @reactive.event(input.doc_cluster_no_sel)
    def _():
        @render.data_frame
        def doc_cluster_top_feat():

            clust_sel_no_now=int(input.doc_cluster_no_sel().split("_")[2])

            #model_doc
            #feat_doc


            data_table={"Token":feat_doc(),"Weight":np.array(model_doc().coef()[clust_sel_no_now,:]).flatten()}
            data_table=pd.DataFrame(data_table)
            data_table=data_table.sort_values(by='Weight', ascending=False)
            return(data_table)

    @reactive.effect
    @reactive.event(input.pop_dif_metrics_button_doc)
    def _():
        doc_topic_dists = np.array(model_doc().basis())
        # doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)

        if input.pop_dif_weight_option_doc() == "max_weight":
            topic_pop_metrics, topic_dif_metrics = pop_dif_fun(
                main_data(), doc_topic_dists, method="max_weight")
        elif input.pop_dif_weight_option_doc() == "all_weight":
            topic_pop_metrics, topic_dif_metrics = pop_dif_fun(
                main_data(), doc_topic_dists, method="all_weight")
        elif input.pop_dif_weight_option_doc() == "corr_option":
            topic_pop_metrics, topic_dif_metrics = pop_dif_fun(
                main_data(), doc_topic_dists, method="corr_option")

        doc_pop_metrics_react.set(topic_pop_metrics)
        doc_dif_metrics_react.set(topic_dif_metrics)

        # @render.data_frame

        @render.table
        def pop_metrics_table_doc():
            # return render.DataGrid(pd.DataFrame(topic_pop_metrics) )#, selection_mode="rows"
            return pd.DataFrame(topic_pop_metrics)

        @render.table
        def dif_metrics_table_doc():
            return pd.DataFrame(topic_dif_metrics)

    @reactive.effect
    @reactive.event(input.doc2d_metric_button)
    def _():
        # topic_2dmetrics_1
        # ,choices={"views_opt":"Average Views","score_opt":"Average Score","comm_opt":"Average Comments","answer_opt":"Average Answers","answer_views_opt":"Average Answers / Views","acc_opt":"Weight with accepted answers"}))
        # topic_pop_metrics = {"Average Views": [], "Average Score": [], #"Average Favorizations": [],"Average Comments": []}
        # topic_dif_metrics = {"Average Answers": [], "Average Answers / Average Views": [],"Weight with accepted answers": []}
        df = {}
        if input.doc_2dmetrics_1() == "views_opt":
            df["Weight Views"] = doc_pop_metrics_react()["Weight Views"]
            x_val = "Weight Views"
        elif input.doc_2dmetrics_1() == "score_opt":
            df["Weight Score"] = doc_pop_metrics_react()["Weight Score"]
            x_val = "Weight Score"
        elif input.doc_2dmetrics_1() == "comm_opt":
            df["Weight Comments"] = doc_pop_metrics_react()["Weight Comments"]
            x_val = "Weight Comments"
        elif input.doc_2dmetrics_1() == "answer_opt":
            df["Weight Answers"] = doc_dif_metrics_react()["Weight Answers"]
            x_val = "Weight Answers"
        elif input.doc_2dmetrics_1() == "answer_views_opt":
            df["Weight Answers / Weight Views"] = doc_dif_metrics_react()["Weight Answers / Weight Views"]
            x_val = "Weight Answers / Weight Views"
        elif input.doc_2dmetrics_1() == "acc_opt":
            df["Weight with accepted answers"] = doc_dif_metrics_react()["Weight with accepted answers"]
            x_val = "Weight with accepted answers"
        elif input.doc_2dmetrics_1() == "poppc_opt":
            df["Popularity Principal component"] = doc_pop_metrics_react()["Popularity Principal component"]
            x_val = "Popularity Principal component"
        elif input.doc_2dmetrics_1() == "difpc_opt":
            df["Difficulty Principal component"] = doc_dif_metrics_react()["Difficulty Principal component"]
            x_val = "Difficulty Principal component"

        if input.doc_2dmetrics_2() == "views_opt":
            df["Weight Views"] = doc_pop_metrics_react()["Weight Views"]
            y_val = "Weight Views"
        elif input.doc_2dmetrics_2() == "score_opt":
            df["Weight Score"] = doc_pop_metrics_react()["Weight Score"]
            y_val = "Weight Score"
        elif input.doc_2dmetrics_2() == "comm_opt":
            df["Weight Comments"] = doc_pop_metrics_react()["Weight Comments"]
            y_val = "Weight Comments"
        elif input.doc_2dmetrics_2() == "answer_opt":
            df["Weight Answers"] = doc_dif_metrics_react()["Weight Answers"]
            y_val = "Weight Answers"
        elif input.doc_2dmetrics_2() == "answer_views_opt":
            df["Weight Answers / Weight Views"] = doc_dif_metrics_react()["Weight Answers / Weight Views"]
            y_val = "Weight Answers / Weight Views"
        elif input.doc_2dmetrics_2() == "acc_opt":
            df["Weight with accepted answers"] = doc_dif_metrics_react()["Weight with accepted answers"]
            y_val = "Weight with accepted answers"
        elif input.doc_2dmetrics_2() == "poppc_opt":
            df["Popularity Principal component"] = doc_pop_metrics_react()["Popularity Principal component"]
            y_val = "Popularity Principal component"
        elif input.doc_2dmetrics_2() == "difpc_opt":
            df["Difficulty Principal component"] = doc_dif_metrics_react()["Difficulty Principal component"]
            y_val = "Difficulty Principal component"

        text_list = []
        for i in range(no_clust_doc()):
            text_list.append(f"Cluster {i}")

        fig = px.scatter(df, x=x_val, y=y_val,
                         text=text_list)  # , color="species" , size='petal_length' , hover_data=['petal_width']

        @render_widget
        def doc2d_vis_widget():

            return (fig)

    @reactive.effect
    @reactive.event(input.growth_metrics_button_doc)
    def _():
        doc_topic_dists = np.array(model_doc().basis())

        if input.doc_growth_weight_option() == "all_weight":
            topic_growth, topic_growth_month = calc_growth_fun(
                main_data(), doc_topic_dists, method="all_weight",
                label="Cluster")
        elif input.doc_growth_weight_option() == "max_weight":
            topic_growth, topic_growth_month = calc_growth_fun(
                main_data(), doc_topic_dists, method="max_weight",
                label="Cluster")

        doc_growth_react.set(topic_growth)
        doc_growth_month_react.set(topic_growth_month)

    @render_widget
    def doc_growth_year_vis():
        # Create the plot
        # print(selected_topic_now)
        if input.doc_model_growth_my() == "per_year":
            fig = go.Figure()
            palette = list(np.random.choice(range(256), size=(no_clust_doc(), 3)))

            for selected_topic_now in range(no_clust_doc()):
                color_now = palette[selected_topic_now]
                color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                doc_growth_react()[f"Cluster {selected_topic_now}"] = OrderedDict(
                    sorted(doc_growth_react()[f"Cluster {selected_topic_now}"].items()))
                fig.add_trace(go.Scatter(
                    x=list(doc_growth_react()[f"Cluster {selected_topic_now}"].keys()),
                    y=list(doc_growth_react()[f"Cluster {selected_topic_now}"].values()),
                    mode='lines+markers',
                    name=f"Cluster {selected_topic_now}",
                    line=dict(color=color_now),
                    marker=dict(color=color_now, size=8)
                ))

            # Add titles and labels
            fig.update_layout(
                title='Cluster Weights Over Years',
                xaxis_title='Year',
                yaxis_title='Weight',
                template='plotly_white'
            )
            return (fig)
        elif input.doc_model_growth_my() == "per_month":
            fig = go.Figure()
            palette = list(np.random.choice(range(256), size=(no_clust_doc(), 3)))

            for selected_topic_now in range(no_clust_doc()):
                color_now = palette[selected_topic_now]
                color_now = f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

                doc_growth_month_react()[f"Cluster {selected_topic_now}"] = OrderedDict(
                    sorted(doc_growth_month_react()[f"Cluster {selected_topic_now}"].items()))
                fig.add_trace(go.Scatter(
                    x=list(doc_growth_month_react()[f"Cluster {selected_topic_now}"].keys()),
                    y=list(doc_growth_month_react()[f"Cluster {selected_topic_now}"].values()),
                    mode='lines+markers',
                    name=f"Cluster {selected_topic_now}",
                    line=dict(color=color_now),
                    marker=dict(color=color_now, size=8)
                ))

            # Add titles and labels
            fig.update_layout(
                title='Cluster Weights Over Months',
                xaxis_title='Month',
                yaxis_title='Weight',
                template='plotly_white'
            )
            return (fig)

    @reactive.effect
    @reactive.event(input.doc_reg_button)
    def _():

        doc_topic_dists = np.array(model_doc().basis())
        #{"views_opt": "Views", "score_opt": "Score", "comm_opt":"Comments",
        #"answer_opt": "Answers", "ans_view_opt":"Answers / Views","acc_opt":"Has accepted answer"}


        df_table=reg_model_res(main_data(),doc_topic_dists,output_opt=input.doc_reg_output(),model_opt=input.doc_reg_opt(),label="Topic")



        @render.table
        def doc_reg_table():
            return(pd.DataFrame(df_table))


app_shiny_now= App(shiny_ui, shiny_server)


if __name__ == '__main__':
    run_app(app_shiny_now)






