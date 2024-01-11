7. Recommendations for Non-Target Structure Nomenclature
7.1 Approach
TG-263 defined the following set of guiding principles for creating structure names. As new structures
are added, following these principles ensures names that are operable with current vended systems and
consistent in structure. This enables the use of computer algorithms to parse names.
The primary objective in defining a nomenclature is to reduce variability in naming. Variation is
the principle barrier to developing automated solutions for accurate extraction, exchange, and processing
of data. Variation in naming occurs over time between individuals and among institutions and vendors.
The second objective for a nomenclature is straightforward adoption into current practice. For
example, the use of just three hexadecimal characters would enable numeric coding of 4096 structures,
leaving ample room to encode other details about the structures and to also be language neutral.
However, proposing that users label the brain as “06E” instead of “Brain” would fail, utterly. To succeed
in reducing data variability while being practical, there were a few situations where it was necessary
to sacrifice internal consistency or strict adherence to a set of ideals in order to define a pragmatic
schema.

7.2 Guiding Principles for Non-Target Nomenclature
1. All structure names are limited to 16 characters or fewer to ensure compatibility with a majority
of vended systems.
2. All structure names must resolve to unique values, independent of capitalization. This ensures
that systems with case-insensitive formats do not result in overlapping definitions.
3. Compound structures are identified using the plural, i.e., the name ends with an ‘s’ or an ‘i’ as
appropriate on the root structure name (e.g., Lungs, Kidneys, Hippocampi, LNs (for all lymph
nodes), Ribs_L.)
4. The first character of each structure category is capitalized (e.g., Femur_Head, Ears_External).
5. No spaces are used.
6. An underscore character is used to separate categorizations (e.g., Bowel_Bag).
7. Spatial categorizations for the primary name are always located at the end of the string following
an underscore character (e.g., Lung_L, Lung_LUL, Lung_RLL, OpticNrv_PRV03_L):
a. L for left
b. R for Right
c. A for Anterior
d. P for Posterior
e. I for Inferior
f. S for Superior
g. RUL, RLL, RML for right upper, lower and middle lobe
h. LUL, LLL for left upper and lower lobe
i. NAdj for non-adjacent
j. Dist for distal, Prox for proximal
8. A consistent root structure name is used for all substructures (e.g., SeminalVes and SeminalVes_
Dist have a consistent root structure name, but SeminalVesicle and SemVes_Dist do
not have a consistent root structure name).
9. Standard category roots are used for structures distributed throughout the body:
a. A for artery (e.g., A_Aorta, A_Carotid)
b. V for vein (e.g., V_Portal, V_Pulmonary)
c. LN for lymph node (e.g., LN_Ax_L1, LN_IMN)
d. CN for cranial nerve (e.g., CN_IX_L, CN_XII_R)
e. Glnd for glandular structure (e.g., Glnd_Submand)
f. Bone (e.g., Bone_Hyoid, Bone_Pelvic)
g. Musc for muscle (e.g., Musc_Masseter, Musc_Sclmast_L)
h. Spc for Space (e.g., Spc_Bowel, Spc_Retrophar_L)
i. VB for vertebral body
j. Sinus for sinus (e.g., Sinus_Frontal, Sinus_Maxillary)
10. Planning organ at risk volumes (PRV) are indicated with PRV following the main structure
separated by an underscore (e.g., Brainstem_PRV). Optionally, the uniform expansion used to
form the PRV from the main structure in millimeters is indicated with two numerals (e.g., SpinalCord_
PRV05, Brainstem_PRV03) unless the result exceeds the character limit. For example,
OpticChiasm_PRV03 is 17 characters and may be truncated to OpticChiasm_PRV3.
11. Partial structures are designated by appending a ‘~’ character to the root name (e.g., Brain~,
Lung~_L). This designator should be used to ensure a contoured structure is not misinterpreted
as a whole organ when such a misinterpretation could have clinical implications (typically
parallel organs). A use case example is a CT scan not long enough to include the full
lung volumes, for which Lungs~ indicates the contoured pair of lungs is only a portion of the
complete structure.
12. If a custom qualifier string is to be used, it is placed at the end after a ‘^’ character (e.g.,
Lungs^Ex).
13. Establish a Primary and a Reverse Order name for each structure.
a. Primary Name. Reading left to right, the structure categorization proceeds from general to
specific with laterality on the end. As a result, alphabetically sorted structure names produce
a list that is grouped by organ (e.g., Kidney_R, Kidney_Cortex_L, Kidney_Hilum_
R). The Primary Name is recommended as the standard choice.
b. Reverse Order Name. Reverse the order of Primary Name. Some vended systems allow
longer strings but have displays that default to show fewer than 16 characters. The Reverse Order Name increases the likelihood that sufficient information can be displayed to safely identify the correct structure. For example, R_Hilum_Kidney would display as R_Hilum_Ki if the vendor’s report only showed the first 10 characters. It is suggested that Reverse Order Name be limited to situations where vendor system constraints prevent safe use of Primary Name.
14. Camel case (a compound word where each word starts with a capital letter and there is no
space between words such as CamelCase) is only used when a structure name implies two
concepts, but the concepts do not appear as distinct categories in common usage (e.g., CaudaEquina
instead of Cauda_Equina) because there are not several examples of Cauda_xxxxx.
Camel case names for primary and reverse order names are identical.
15. Structures that are not used for dose evaluation (e.g., optimization structures, high/low dose
regions) should be prefixed with a ‘z’ or ‘_’ character so that an alphabetical sort groups them
away from structures that are used for dose evaluation (e.g., zPTVopt). Selection of ‘z’ to designate
dose evaluation structures is suggested.


4.3 Nomenclature for Target Structures
Target structures showed wider variation in nomenclature approaches than non-target structures. Various
combinations of prefixes and suffixes for ICRU-defined targets (GTV, CTV, ITV, PTV) as well as
tumor bed volumes (TBV), internal gross target volumes (IGTV), and internal clinical target volumes
(ICTV) were used to define the target location, target number, structure type, dose delivered, revision
number, identity of person contouring, etc. Variations in capitalization and element separations were
similar to normal structures. Thirteen institutions reported standardized nomenclatures for targets, and
examples of different nomenclatures are listed in Table 2.
4.4 Derived and Planning Structures
Derivative structures are formed from target or non-target structures, typically using Boolean operations,
e.g., intersection (x AND y), combination (x OR y), subtraction (x AND NOT y), and margins
(x+1.0). Five institutions indicated that nomenclatures for derivative structures were used to define
conditions for evaluating the dose distribution (e.g., OAR contour excluding PTV). Variations in several
structures were common (e.g., body-ptv, PTV_EVAL, eval_PTV), but wide variation was noted
for structures involving multiple concepts (e.g., NS_Brain-PTVs, optCTV-N2R5L_MRT1_ex-3600-
v12).
Institutions indicated that structures were frequently created as a tool for dose optimization as
opposed to dose evaluation. For example, an optimization structure created from a copy of the PTV
structure with a Boolean operation excluding critical OAR structures from it to reflect dose compromises
in plan optimization is routinely created by multiple institutions. However, naming conventions
for such structures varied among members (e.g., modPTV, opt PTV, PTV_OPT). Although clinical
flow may improve with minimal constraints on naming of dose sculpting structures, members noted
that these structures can present a safety issue if they are confused with the structures used for dose
evaluation (e.g., PTV, PTVHot, PTVCold, Ring, DLA.) To minimize possible usage for the wrong purpose, several institutions selected a single character (e.g., ‘z’ or ‘_’) that was uniformly applied as a prefix to those structures. This prefix ensured that in an alphabetical sort they appeared at the end or
beginning of the list (e.g., PTV, zPTVHot, zPTVCold, zRing, zDLA). Selection of z as a prefix is
suggested.

8.1 Approach
Surveys of member responses for target naming strategies revealed that clinics use a very complex set
of concepts: ICRU and other types, target classifiers for primary and nodal volumes, enumeration of
volumes when there are several structures, dose, basis structures, imaging modality used to create, etc.
Clinics did not attempt to represent all concepts, but selected those few considered most important
to their process. Within an individual clinic, different naming strategies could be used for different
treatment sites or physicians.
Task Group 263 determined that it could not come to consensus to define a single standard for all
use cases and clinics that spanned the numerous concepts for a target name and also meet character
string constraints. However, TG-263 did establish a set of guiding principles to specify where and how
a concept should appear if it is represented in the target name. Therefore, Task Group 263 established
a set of guiding principles for target nomenclature.
This approach enables construction of computer algorithms to parse the names or to automatically
create names based on concepts selected by users. Users choose the supplemental information to
incorporate into target names, and these guiding principles ensure that computer programs can recognize
these names for quality and research endeavors. While these principles accommodate the majority of encountered names, they cannot accommodate all. TG-263 recommends using the ‘^’ character
to designate supplemental information not incorporated in the current guidelines.

8.2 Guiding Principles for Target Nomenclature
1. The first set of characters must be one of the allowed target types:
• GTV
• CTV
• ITV
• IGTV (Internal Gross Target Volume—gross disease with margin for motion)
• ICTV (Internal Clinical Target Volume—clinical disease with margin for motion)
• PTV
• PTV! for low-dose PTV volumes that exclude overlapping high-dose volumes (See the section
discussing segmented vs non-segmented PTVs.)
2. If a target classifier is used, place the target classifier after the target type with no spaces.
Allowed target classifiers are listed below:
• n: nodal (e.g., PTVn)
• p: primary (e.g., GTVp)
• sb: surgical bed (e.g., CTVsb)
• par: parenchyma (e.g., GTVpar)
• v:venous thrombosis (e.g., CTVv)
• vas: vascular (e.g., CTVvas)
3. If multiple spatially distinct targets are indicated, then Arabic numerals are used after the target
type + classifier (e.g., PTV1, PTV2, GTVp1, GTVp2).)
4. If designation of the imaging modality and sequential order in the image set need recording
for adaptive therapy, then the nomenclature follows the type/classifier/enumerator with an
underscore and then the image modality type (CT, PT, MR, SP) and number of the image in
the sequence (e.g., PTVp1_CT1PT1, GTV_CT2).)
5. If structure indicators are used, they follow the type/classifier/enumerator/imaging with an
underscore prefix and are values from the approved structure nomenclature list, (e.g.,
CTV_A_Aorta, CTV_A_Celiac, GTV_Preop, PTV_Boost, PTV_Eval, PTV_MR2_Prostate).
6. If dose is indicated, the dose is placed at the end of the target string prefixed with an underscore
character.
• The task group strongly recommends using relative dose levels instead of specifying
physical dose
◦ High (e.g., PTV_High, CTV_High, GTV_High)
◦ Low (e.g., PTV_Low, CTV_Low, GTV_Low)
◦ Mid: (e.g., PTV_Mid, CTV_Mid, GTV_Mid)
◦ Mid+2-digit enumerator: allows specification of more than three relative dose levels (e.g.,
PTV_Low, PTV_Mid01, PTV_Mid02, PTV_Mid03, PTV_High). Lower numbers correspond
to lower dose values.
• If numeric values for the physical dose must be used, then specification of the numeric value
of the dose in units of cGy is strongly recommended (e.g., PTV_5040).
• If numeric values for physical dose must be used and these must be specified in units of Gy,
then ‘Gy’ should be appended to the numeric value of the dose (e.g., PTV_50.4Gy). For systems
that do not allow use of a period, the ‘p’ character should be substituted (e.g.,
PTV_50p4Gy)
7. If the dose indicated must reflect the number of fractions used to reach the total dose, then the
numeric values of dose per fraction in cGy, or in Gy with the unit specifier, and number of
fractions separated by an “x” character are added at the end (e.g., PTV_Liver_2000x3 or
PTV_Liver_20Gyx3).
8. If the structure is cropped back from the external contour for the patient, then the quantity of
cropping by “-xx” millimeters is placed at the end of the target string. The cropping length follows
the dose indicator, with the amount of cropping indicated by xx millimeters (e.g.,
PTV_Eval_7000-08, PTV-03, CTVp2-05).
9. If a custom qualifier string is used, the custom qualifier is placed at the end after a ‘^’ character
(e.g., PTV^Physician1, GTV_Liver^ICG).)
10. If it is not possible to follow the guidelines and remain within the 16-character limit, then preserve
the relative ordering but remove underscore characters, progressing from left to right as
needed to meet the limit (e.g PTVLiverR_2000x3.) This last resort scenario undermines the
use of automated tools.

