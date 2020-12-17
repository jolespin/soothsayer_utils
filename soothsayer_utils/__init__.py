# ==============
# Soothsayer Utils
# ==============
# Utility functions for Soothsayer
# ------------------------------------
# GitHub: https://github.com/jolespin/soothsayer_utils
# PyPI: https://pypi.org/project/soothsayer_utils/
# ------------------------------------
# =======
# Contact
# =======
# Producer: Josh L. Espinoza
# Contact: jespinoz@jcvi.org, jol.espinoz@gmail.com
# Google Scholar: https://scholar.google.com/citations?user=r9y1tTQAAAAJ&hl
# =======
# License BSD-3
# =======
# https://opensource.org/licenses/BSD-3-Clause
#
# Copyright 2018 Josh L. Espinoza
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# =======
# Version
# =======
__version__= "2020.12.15"
__author__ = "Josh L. Espinoza"
__email__ = "jespinoz@jcvi.org, jol.espinoz@gmail.com"
__url__ = "https://github.com/jolespin/soothsayer_utils"
__license__ = "BSD-3"
__developmental__ = True

# =======
# Direct Exports
# =======
__functions__ = [
 'read_star_log', 'read_hmmer','read_gtf_gff_base', 'assert_acceptable_arguments', 'boolean', 
 'consecutive_replace', 'contains',  'dict_build', 'dict_collapse', 'dict_expand', 'dict_fill', 
 'dict_filter', 'dict_reverse', 'dict_tree',  'flatten','format_memory', 'format_duration', 
 'format_header', 'format_path', 'fragment',  'get_timestamp', 'get_unique_identifier', 'hash_kmer', 
 'infer_compression', 'is_all_same_type', 'is_dict', 'is_dict_like', 'is_file_like', 'is_function', 
 'is_in_namespace', 'is_nonstring_iterable', 'is_number', 'is_path_like', 'is_query_class',"is_symmetrical", "is_graph",
 'iterable_depth', 'join_as_strings',  'pad_left',   'pv', 'range_like', 'read_blast', 'read_dataframe',
  'read_ebi_sample_metadata', 'read_fasta', 'read_from_clipboard', 'read_gff3', 'read_gtf', 'read_ncbi_xml', 
  'read_object', 'read_script_as_module', 'read_url', 'reverse_complement', 'to_precision', 
   'write_dataframe', 'write_fasta', 'write_object',"get_file_object","read_textfile","check_packages","get_iris_data",
   "get_color_midpoint","remove_pairwise_nan","format_pairwise","dict_py_to_bash",
]
__classes__ = ["Suppress"]

__all__ = sorted(__functions__ + __classes__)

from .soothsayer_utils import *
