"""Closed list of the 142 SEAD-4 adjudicative-guideline conditions.

Vendored, with attribution, from sead4_llm/config/guidelines.py in the doha
repo. The wording is the National Security Adjudicative Guidelines issued
with Security Executive Agent Directive 4 (June 8, 2017). Line wraps in
that file are collapsed here; the words are not edited. Severity scales
and Bond Amendment notes from the same file are not conditions, so they
are not part of this list.

The rule-based extractor and the LLM extractor both have to select from
this list. `canonical` returns a code only when it is one of these 142, so
a pattern match or a model answer cannot keep a code it invented.
"""
from __future__ import annotations

import re

from src import config

# letter -> name, concern paragraph, disqualifying conditions, mitigating conditions.
# Condition codes are the canonical form used in the decisions: "AG ¶ 19(a)".
GUIDELINES: dict[str, dict] = {
    "A": {
        "name": "Allegiance to the United States",
        "concern": "The willingness to safeguard classified or sensitive information is in doubt if there is any reason to suspect an individual's allegiance to the United States. There is no positive test for allegiance, but there are negative indicators. These include participation in or support for acts against the United States or placing the welfare or interests of another country above those of the United States.",
        "disqualifiers": [
            {
                "code": "AG ¶ 4(a)",
                "text": "involvement in, support of, training to commit, or advocacy of any act of sabotage, espionage, treason, terrorism, or sedition against the United States"
            },
            {
                "code": "AG ¶ 4(b)",
                "text": "association or sympathy with persons who are attempting to commit, or who are committing, any of the above acts"
            },
            {
                "code": "AG ¶ 4(c)",
                "text": "association or sympathy with persons or organizations that advocate, threaten, or use force or violence, or use any other illegal or unconstitutional means, in an effort to: (1) overthrow or influence the U.S. Government or any state or local government; (2) prevent Federal, state, or local government personnel from performing their official duties; (3) gain retribution for perceived wrongs caused by the Federal, state, or local government; and (4) prevent others from exercising their rights under the Constitution or laws of the United States"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 5(a)",
                "text": "the individual was unaware of the unlawful aims of the individual or organization and severed ties upon learning of these"
            },
            {
                "code": "AG ¶ 5(b)",
                "text": "the individual's involvement was humanitarian and permitted under U.S. law"
            },
            {
                "code": "AG ¶ 5(c)",
                "text": "involvement in the above activities occurred for only a short period of time and was attributable to curiosity or academic interest"
            },
            {
                "code": "AG ¶ 5(d)",
                "text": "the involvement or association with such activities occurred under such unusual circumstances, or so much time has elapsed, that it is unlikely to recur and does not cast doubt on the individual's current reliability, trustworthiness, or allegiance"
            }
        ]
    },
    "B": {
        "name": "Foreign Influence",
        "concern": "Foreign contacts and interests, including, but not limited to, business, financial, and property interests, are a national security concern if they result in divided allegiance. They may also be a national security concern if they create circumstances in which the individual may be manipulated or induced to help a foreign person, group, organization, or government in a way inconsistent with U.S. interests or otherwise made vulnerable to pressure or coercion by any foreign interest.",
        "disqualifiers": [
            {
                "code": "AG ¶ 7(a)",
                "text": "contact, regardless of method, with a foreign family member, business or professional associate, friend, or other person who is a citizen of or resident in a foreign country if that contact creates a heightened risk of foreign exploitation, inducement, manipulation, pressure, or coercion"
            },
            {
                "code": "AG ¶ 7(b)",
                "text": "connections to a foreign person, group, government, or country that create a potential conflict of interest between the individual's obligation to protect classified or sensitive information or technology and the individual's desire to help a foreign person, group, or country by providing that information or technology"
            },
            {
                "code": "AG ¶ 7(c)",
                "text": "failure to report or fully disclose, when required, association with a foreign person, group, government, or country"
            },
            {
                "code": "AG ¶ 7(d)",
                "text": "counterintelligence information, whether classified or unclassified, that indicates the individual's access to classified information or eligibility for a sensitive position may involve unacceptable risk to national security"
            },
            {
                "code": "AG ¶ 7(e)",
                "text": "shared living quarters with a person or persons, regardless of citizenship status, if that relationship creates a heightened risk of foreign inducement, manipulation, pressure, or coercion"
            },
            {
                "code": "AG ¶ 7(f)",
                "text": "substantial business, financial, or property interests in a foreign country, or in any foreign owned or foreign-operated business that could subject the individual to a heightened risk of foreign influence or exploitation or personal conflict of interest"
            },
            {
                "code": "AG ¶ 7(g)",
                "text": "unauthorized association with a suspected or known agent, associate, or employee of a foreign intelligence entity"
            },
            {
                "code": "AG ¶ 7(h)",
                "text": "indications that representatives or nationals from a foreign country are acting to increase the vulnerability of the individual to possible future exploitation, inducement, manipulation, pressure, or coercion"
            },
            {
                "code": "AG ¶ 7(i)",
                "text": "conduct, especially while traveling or residing outside the U.S., that may make the individual vulnerable to exploitation, pressure, or coercion by a foreign person, group, government, or country"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 8(a)",
                "text": "the nature of the relationships with foreign persons, the country in which these persons are located, or the positions or activities of those persons in that country are such that it is unlikely the individual will be placed in a position of having to choose between the interests of a foreign individual, group, organization, or government and the interests of the United States"
            },
            {
                "code": "AG ¶ 8(b)",
                "text": "there is no conflict of interest, either because the individual's sense of loyalty or obligation to the foreign person, or allegiance to the group, government, or country is so minimal, or the individual has such deep and longstanding relationships and loyalties in the United States, that the individual can be expected to resolve any conflict of interest in favor of the U.S. interest"
            },
            {
                "code": "AG ¶ 8(c)",
                "text": "contact or communication with foreign citizens is so casual and infrequent that there is little likelihood that it could create a risk for foreign influence or exploitation"
            },
            {
                "code": "AG ¶ 8(d)",
                "text": "the foreign contacts and activities are on U.S. Government business or are approved by the agency head or designee"
            },
            {
                "code": "AG ¶ 8(e)",
                "text": "the individual has promptly complied with existing agency requirements regarding the reporting of contacts, requests, or threats from persons, groups, or organizations from a foreign country"
            },
            {
                "code": "AG ¶ 8(f)",
                "text": "the value or routine nature of the foreign business, financial, or property interests is such that they are unlikely to result in a conflict and could not be used effectively to influence, manipulate, or pressure the individual"
            }
        ]
    },
    "C": {
        "name": "Foreign Preference",
        "concern": "When an individual acts in such a way as to indicate a preference for a foreign country over the United States, then he or she may provide information or make decisions that are harmful to the interests of the United States. Foreign involvement raises concerns about an individual's judgment, reliability, and trustworthiness when it is in conflict with U.S. national interests or when the individual acts to conceal it.",
        "disqualifiers": [
            {
                "code": "AG ¶ 10(a)",
                "text": "applying for and/or acquiring citizenship in any other country"
            },
            {
                "code": "AG ¶ 10(b)",
                "text": "failure to report, or fully disclose when required, to an appropriate security official, the possession of a passport or identity card issued by any country other than the United States"
            },
            {
                "code": "AG ¶ 10(c)",
                "text": "failure to use a U.S. passport when entering or exiting the U.S."
            },
            {
                "code": "AG ¶ 10(d)",
                "text": "participation in foreign activities, including but not limited to: (1) assuming or attempting to assume any type of employment, position, or political office in a foreign government or military organization; and (2) otherwise acting to serve the interests of a foreign person, group, organization, or government in any way that conflicts with U.S. national security interests"
            },
            {
                "code": "AG ¶ 10(e)",
                "text": "using foreign citizenship to protect financial or business interests in another country in violation of U.S. law"
            },
            {
                "code": "AG ¶ 10(f)",
                "text": "an act of expatriation from the United States such as declaration of intent to renounce U.S. citizenship, whether through words or actions"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 11(a)",
                "text": "the foreign citizenship is not in conflict with U.S. national security interests"
            },
            {
                "code": "AG ¶ 11(b)",
                "text": "dual citizenship is based solely on parental citizenship or birth in a foreign country, and there is no evidence of foreign preference"
            },
            {
                "code": "AG ¶ 11(c)",
                "text": "the individual has expressed a willingness to renounce the foreign citizenship that is in conflict with U.S. national security interests"
            },
            {
                "code": "AG ¶ 11(d)",
                "text": "the exercise of the rights, privileges, or obligations of foreign citizenship occurred before the individual became a U.S. citizen"
            },
            {
                "code": "AG ¶ 11(e)",
                "text": "the exercise of the entitlements or benefits of foreign citizenship do not present a national security concern"
            },
            {
                "code": "AG ¶ 11(f)",
                "text": "the foreign preference, if detected, involves a foreign country, entity, or association that poses a low national security risk"
            }
        ]
    },
    "D": {
        "name": "Sexual Behavior",
        "concern": "Sexual behavior that involves a criminal offense; reflects a lack of judgment or discretion; or may subject the individual to undue influence of coercion, exploitation, or duress. These issues, together or individually, may raise questions about an individual's judgment, reliability, trustworthiness, and ability to protect classified or sensitive information.",
        "disqualifiers": [
            {
                "code": "AG ¶ 13(a)",
                "text": "sexual behavior of a criminal nature, whether or not the individual has been prosecuted"
            },
            {
                "code": "AG ¶ 13(b)",
                "text": "a pattern of compulsive, self-destructive, or high-risk sexual behavior that the individual is unable to stop"
            },
            {
                "code": "AG ¶ 13(c)",
                "text": "sexual behavior that causes an individual to be vulnerable to coercion, exploitation, or duress"
            },
            {
                "code": "AG ¶ 13(d)",
                "text": "sexual behavior of a public nature or that reflects lack of discretion or judgment"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 14(a)",
                "text": "the behavior occurred prior to or during adolescence and there is no evidence of subsequent conduct of a similar nature"
            },
            {
                "code": "AG ¶ 14(b)",
                "text": "the sexual behavior happened so long ago, so infrequently, or under such unusual circumstances, that it is unlikely to recur and does not cast doubt on the individual's current reliability, trustworthiness, or judgment"
            },
            {
                "code": "AG ¶ 14(c)",
                "text": "the behavior no longer serves as a basis for coercion, exploitation, or duress"
            },
            {
                "code": "AG ¶ 14(d)",
                "text": "the sexual behavior is strictly private, consensual, and discreet"
            },
            {
                "code": "AG ¶ 14(e)",
                "text": "the individual has successfully completed an appropriate program of treatment, or is currently enrolled in one, has demonstrated ongoing and consistent compliance with the treatment plan, and/or has received a favorable prognosis from a qualified mental health professional indicating the behavior is readily controllable with treatment"
            }
        ]
    },
    "E": {
        "name": "Personal Conduct",
        "concern": "Conduct involving questionable judgment, lack of candor, dishonesty, or unwillingness to comply with rules and regulations can raise questions about an individual's reliability, trustworthiness, and ability to protect classified or sensitive information. Of special interest is any failure to cooperate or provide truthful and candid answers during national security investigative or adjudicative processes.",
        "disqualifiers": [
            {
                "code": "AG ¶ 16(a)",
                "text": "deliberate omission, concealment, or falsification of relevant facts from any personnel security questionnaire, personal history statement, or similar form used to conduct investigations, determine employment qualifications, award benefits or status, determine national security eligibility or trustworthiness, or award fiduciary responsibilities"
            },
            {
                "code": "AG ¶ 16(b)",
                "text": "deliberately providing false or misleading information; or concealing or omitting information, concerning relevant facts to an employer, investigator, security official, competent medical or mental health professional involved in making a recommendation relevant to a national security eligibility determination, or other official government representative"
            },
            {
                "code": "AG ¶ 16(c)",
                "text": "credible adverse information in several adjudicative issue areas that is not sufficient for an adverse determination under any other single guideline, but which, when considered as a whole, supports a whole-person assessment of questionable judgment, untrustworthiness, unreliability, lack of candor, unwillingness to comply with rules and regulations, or other characteristics indicating that the individual may not properly safeguard classified or sensitive information"
            },
            {
                "code": "AG ¶ 16(d)",
                "text": "credible adverse information that is not explicitly covered under any other guideline and may not be sufficient by itself for an adverse determination, but which, when combined with all available information, supports a whole-person assessment of questionable judgment, untrustworthiness, unreliability, lack of candor, unwillingness to comply with rules and regulations"
            },
            {
                "code": "AG ¶ 16(e)",
                "text": "personal conduct, or concealment of information about one's conduct, that creates a vulnerability to exploitation, manipulation, or duress by a foreign intelligence entity or other individual or group"
            },
            {
                "code": "AG ¶ 16(f)",
                "text": "violation of a written or recorded commitment made by the individual to the employer as a condition of employment"
            },
            {
                "code": "AG ¶ 16(g)",
                "text": "association with persons involved in criminal activity"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 17(a)",
                "text": "the individual made prompt, good-faith efforts to correct the omission, concealment, or falsification before being confronted with the facts"
            },
            {
                "code": "AG ¶ 17(b)",
                "text": "the refusal or failure to cooperate, omission, or concealment was caused or significantly contributed to by advice of legal counsel or of a person with professional responsibilities for advising or instructing the individual specifically concerning security processes"
            },
            {
                "code": "AG ¶ 17(c)",
                "text": "the offense is so minor, or so much time has passed, or the behavior is so infrequent, or it happened under such unique circumstances that it is unlikely to recur and does not cast doubt on the individual's reliability, trustworthiness, or good judgment"
            },
            {
                "code": "AG ¶ 17(d)",
                "text": "the individual has acknowledged the behavior and obtained counseling to change the behavior or taken other positive steps to alleviate the stressors, circumstances, or factors that contributed to untrustworthy, unreliable, or other inappropriate behavior, and such behavior is unlikely to recur"
            },
            {
                "code": "AG ¶ 17(e)",
                "text": "the individual has taken positive steps to reduce or eliminate vulnerability to exploitation, manipulation, or duress"
            },
            {
                "code": "AG ¶ 17(f)",
                "text": "the information was unsubstantiated or from a source of questionable reliability"
            },
            {
                "code": "AG ¶ 17(g)",
                "text": "association with persons involved in criminal activities was unwitting, has ceased, or occurs under circumstances that do not cast doubt upon the individual's reliability, trustworthiness, judgment, or willingness to comply with rules and regulations"
            }
        ]
    },
    "F": {
        "name": "Financial Considerations",
        "concern": "Failure to live within one's means, satisfy debts, and meet financial obligations may indicate poor self-control, lack of judgment, or unwillingness to abide by rules and regulations, all of which can raise questions about an individual's reliability, trustworthiness, and ability to protect classified or sensitive information. Financial distress can also be caused or exacerbated by, and thus can be a possible indicator of, other issues of personnel security concern such as excessive gambling, mental health conditions, substance misuse, or alcohol abuse or dependence.",
        "disqualifiers": [
            {
                "code": "AG ¶ 19(a)",
                "text": "inability to satisfy debts"
            },
            {
                "code": "AG ¶ 19(b)",
                "text": "unwillingness to satisfy debts regardless of the ability to do so"
            },
            {
                "code": "AG ¶ 19(c)",
                "text": "a history of not meeting financial obligations"
            },
            {
                "code": "AG ¶ 19(d)",
                "text": "deceptive or illegal financial practices such as embezzlement, employee theft, check fraud, expense account fraud, mortgage fraud, filing deceptive loan statements and other intentional financial breaches of trust"
            },
            {
                "code": "AG ¶ 19(e)",
                "text": "consistent spending beyond one's means or frivolous or irresponsible spending, which may be indicated by excessive indebtedness, significant negative cash flow, a history of late payments or of non-payment, or other negative financial indicators"
            },
            {
                "code": "AG ¶ 19(f)",
                "text": "failure to file or fraudulently filing annual Federal, state, or local income tax returns or failure to pay annual Federal, state, or local income tax as required"
            },
            {
                "code": "AG ¶ 19(g)",
                "text": "unexplained affluence, as shown by a lifestyle or standard of living, increase in net worth, or money transfers that are inconsistent with known legal sources of income"
            },
            {
                "code": "AG ¶ 19(h)",
                "text": "borrowing money or engaging in significant financial transactions to fund gambling or pay gambling debts"
            },
            {
                "code": "AG ¶ 19(i)",
                "text": "concealing gambling losses, family conflict, or other problems caused by gambling"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 20(a)",
                "text": "the behavior happened so long ago, was so infrequent, or occurred under such circumstances that it is unlikely to recur and does not cast doubt on the individual's current reliability, trustworthiness, or good judgment"
            },
            {
                "code": "AG ¶ 20(b)",
                "text": "the conditions that resulted in the financial problem were largely beyond the person's control (e.g., loss of employment, a business downturn, unexpected medical emergency, a death, divorce or separation, clear victimization by predatory lending practices, or identity theft), and the individual acted responsibly under the circumstances"
            },
            {
                "code": "AG ¶ 20(c)",
                "text": "the individual has received or is receiving financial counseling for the problem from a legitimate and credible source, such as a non-profit credit counseling service, and there are clear indications that the problem is being resolved or is under control"
            },
            {
                "code": "AG ¶ 20(d)",
                "text": "the individual initiated and is adhering to a good-faith effort to repay overdue creditors or otherwise resolve debts"
            },
            {
                "code": "AG ¶ 20(e)",
                "text": "the individual has a reasonable basis to dispute the legitimacy of the past-due debt which is the cause of the problem and provides documented proof to substantiate the basis of the dispute or provides evidence of actions to resolve the issue"
            },
            {
                "code": "AG ¶ 20(f)",
                "text": "the affluence resulted from a legal source of income"
            },
            {
                "code": "AG ¶ 20(g)",
                "text": "the individual has made arrangements with the appropriate tax authority to file or pay the amount owed and is in compliance with those arrangements"
            }
        ]
    },
    "G": {
        "name": "Alcohol Consumption",
        "concern": "Excessive alcohol consumption often leads to the exercise of questionable judgment or the failure to control impulses, and can raise questions about an individual's reliability and trustworthiness.",
        "disqualifiers": [
            {
                "code": "AG ¶ 22(a)",
                "text": "alcohol-related incidents away from work, such as driving while under the influence, fighting, child or spouse abuse, disturbing the peace, or other incidents of concern, regardless of the frequency of the individual's alcohol use or whether the individual has been diagnosed with alcohol use disorder"
            },
            {
                "code": "AG ¶ 22(b)",
                "text": "alcohol-related incidents at work, such as reporting for work or duty in an intoxicated or impaired condition, drinking on the job, or jeopardizing the welfare and safety of others, regardless of whether the individual is diagnosed with alcohol use disorder"
            },
            {
                "code": "AG ¶ 22(c)",
                "text": "habitual or binge consumption of alcohol to the point of impaired judgment, regardless of whether the individual is diagnosed with alcohol use disorder"
            },
            {
                "code": "AG ¶ 22(d)",
                "text": "diagnosis by a duly qualified medical or mental health professional (e.g., physician, clinical psychologist, psychiatrist, or licensed clinical social worker) of alcohol use disorder"
            },
            {
                "code": "AG ¶ 22(e)",
                "text": "the failure to follow treatment advice once diagnosed"
            },
            {
                "code": "AG ¶ 22(f)",
                "text": "alcohol consumption, which is not in accordance with treatment recommendations after a diagnosis of alcohol use disorder"
            },
            {
                "code": "AG ¶ 22(g)",
                "text": "failure to follow any court order regarding alcohol education, evaluation, treatment, or abstinence"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 23(a)",
                "text": "so much time has passed, or the behavior was so infrequent, or it happened under such unusual circumstances that it is unlikely to recur or does not cast doubt on the individual's current reliability, trustworthiness, or judgment"
            },
            {
                "code": "AG ¶ 23(b)",
                "text": "the individual acknowledges his or her pattern of maladaptive alcohol use, provides evidence of actions taken to overcome this problem, and has demonstrated a clear and established pattern of modified consumption or abstinence in accordance with treatment recommendations"
            },
            {
                "code": "AG ¶ 23(c)",
                "text": "the individual is participating in counseling or a treatment program, has no previous history of treatment and relapse, and is making satisfactory progress in a treatment program"
            },
            {
                "code": "AG ¶ 23(d)",
                "text": "the individual has successfully completed a treatment program along with any required aftercare, and has demonstrated a clear and established pattern of modified consumption or abstinence in accordance with treatment recommendations"
            }
        ]
    },
    "H": {
        "name": "Drug Involvement and Substance Misuse",
        "concern": "The illegal use of controlled substances, to include the misuse of prescription and non-prescription drugs, and the use of other substances that cause physical or mental impairment or are used in a manner inconsistent with their intended purpose can raise questions about an individual's reliability and trustworthiness, both because such behavior may lead to physical or psychological impairment and because it raises questions about a person's ability or willingness to comply with laws, rules, and regulations.",
        "disqualifiers": [
            {
                "code": "AG ¶ 25(a)",
                "text": "any substance misuse"
            },
            {
                "code": "AG ¶ 25(b)",
                "text": "testing positive for an illegal drug"
            },
            {
                "code": "AG ¶ 25(c)",
                "text": "illegal possession of a controlled substance, including cultivation, processing, manufacture, purchase, sale, or distribution; or possession of drug paraphernalia"
            },
            {
                "code": "AG ¶ 25(d)",
                "text": "diagnosis by a duly qualified medical or mental health professional (e.g., physician, clinical psychologist, psychiatrist, or licensed clinical social worker) of substance use disorder"
            },
            {
                "code": "AG ¶ 25(e)",
                "text": "failure to successfully complete a drug treatment program prescribed by a duly qualified medical or mental health professional"
            },
            {
                "code": "AG ¶ 25(f)",
                "text": "any illegal drug use while granted access to classified information or holding a sensitive position"
            },
            {
                "code": "AG ¶ 25(g)",
                "text": "expressed intent to continue drug involvement and substance misuse, or failure to clearly and convincingly commit to discontinue such misuse"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 26(a)",
                "text": "the behavior happened so long ago, was so infrequent, or happened under such circumstances that it is unlikely to recur or does not cast doubt on the individual's current reliability, trustworthiness, or good judgment"
            },
            {
                "code": "AG ¶ 26(b)",
                "text": "the individual acknowledges his or her drug involvement and substance misuse, provides evidence of actions taken to overcome this problem, and has established a pattern of abstinence, including, but not limited to: (1) disassociation from drug-using associates and contacts; (2) changing or avoiding the environment where drugs were used; and (3) providing a signed statement of intent to abstain from all drug involvement and substance misuse, acknowledging that any future involvement or misuse is grounds for revocation of national security eligibility"
            },
            {
                "code": "AG ¶ 26(c)",
                "text": "abuse of prescription drugs was after a severe or prolonged illness during which these drugs were prescribed, and abuse has since ended"
            },
            {
                "code": "AG ¶ 26(d)",
                "text": "satisfactory completion of a prescribed drug treatment program, including, but not limited to, rehabilitation and aftercare requirements, without recurrence of abuse, and a favorable prognosis by a duly qualified medical professional"
            }
        ]
    },
    "I": {
        "name": "Psychological Conditions",
        "concern": "Certain emotional, mental, and personality conditions can impair judgment, reliability, or trustworthiness. A formal diagnosis of a disorder is not required for there to be a concern under this guideline. A duly qualified mental health professional employed by, or acceptable to and approved by the U.S. Government, should be consulted when evaluating potentially disqualifying and mitigating information under this guideline.",
        "disqualifiers": [
            {
                "code": "AG ¶ 28(a)",
                "text": "behavior that casts doubt on an individual's judgment, stability, reliability, or trustworthiness, not covered under any other guideline and that may indicate an emotional, mental, or personality condition, including, but not limited to, irresponsible, violent, self-harm, suicidal, paranoid, manipulative, impulsive, chronic lying, deceitful, exploitative, or bizarre behaviors"
            },
            {
                "code": "AG ¶ 28(b)",
                "text": "an opinion by a duly qualified mental health professional that the individual has a condition that may impair judgment, stability, reliability, or trustworthiness"
            },
            {
                "code": "AG ¶ 28(c)",
                "text": "voluntary or involuntary inpatient hospitalization"
            },
            {
                "code": "AG ¶ 28(d)",
                "text": "failure to follow a prescribed treatment plan related to a diagnosed psychological/psychiatric condition that may impair judgment, stability, reliability, or trustworthiness"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 29(a)",
                "text": "the identified condition is readily controllable with treatment, and the individual has demonstrated ongoing and consistent compliance with the treatment plan"
            },
            {
                "code": "AG ¶ 29(b)",
                "text": "the individual has voluntarily entered a counseling or treatment program for a condition that is amenable to treatment, and the individual is currently receiving counseling or treatment with a favorable prognosis by a duly qualified mental health professional"
            },
            {
                "code": "AG ¶ 29(c)",
                "text": "recent opinion by a duly qualified mental health professional employed by, or acceptable to and approved by, the U.S. Government that an individual's previous condition is under control or in remission, and has a low probability of recurrence or exacerbation"
            },
            {
                "code": "AG ¶ 29(d)",
                "text": "the past psychological/psychiatric condition was temporary, the situation has been resolved, and the individual no longer shows indications of emotional instability"
            },
            {
                "code": "AG ¶ 29(e)",
                "text": "there is no indication of a current problem"
            }
        ]
    },
    "J": {
        "name": "Criminal Conduct",
        "concern": "Criminal activity creates doubt about a person's judgment, reliability, and trustworthiness. By its very nature, it calls into question a person's ability or willingness to comply with laws, rules, and regulations.",
        "disqualifiers": [
            {
                "code": "AG ¶ 31(a)",
                "text": "a pattern of minor offenses, any one of which on its own would be unlikely to affect a national security eligibility decision, but which in combination cast doubt on the individual's judgment, reliability, or trustworthiness"
            },
            {
                "code": "AG ¶ 31(b)",
                "text": "evidence (including, but not limited to, a credible allegation, an admission, and matters of official record) of criminal conduct, regardless of whether the individual was formally charged, prosecuted, or convicted"
            },
            {
                "code": "AG ¶ 31(c)",
                "text": "individual is currently on parole or probation"
            },
            {
                "code": "AG ¶ 31(d)",
                "text": "violation or revocation of parole or probation, or failure to complete a court-mandated rehabilitation program"
            },
            {
                "code": "AG ¶ 31(e)",
                "text": "discharge or dismissal from the Armed Forces for reasons less than 'Honorable'"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 32(a)",
                "text": "so much time has elapsed since the criminal behavior happened, or it happened under such unusual circumstances, that it is unlikely to recur and does not cast doubt on the individual's reliability, trustworthiness, or good judgment"
            },
            {
                "code": "AG ¶ 32(b)",
                "text": "the individual was pressured or coerced into committing the act and those pressures are no longer present in the person's life"
            },
            {
                "code": "AG ¶ 32(c)",
                "text": "no reliable evidence to support that the individual committed the offense"
            },
            {
                "code": "AG ¶ 32(d)",
                "text": "there is evidence of successful rehabilitation; including, but not limited to, the passage of time without recurrence of criminal activity, restitution, compliance with the terms of parole or probation, job training or higher education, good employment record, or constructive community involvement"
            }
        ]
    },
    "K": {
        "name": "Handling Protected Information",
        "concern": "Deliberate or negligent failure to comply with rules and regulations for handling protected information—which includes classified and other sensitive government information, and proprietary information—raises doubt about an individual's trustworthiness, judgment, reliability, or willingness and ability to safeguard such information, and is a serious security concern.",
        "disqualifiers": [
            {
                "code": "AG ¶ 34(a)",
                "text": "deliberate or negligent disclosure of protected information to unauthorized persons, including, but not limited to, personal or business contacts, the media, or persons present at seminars, meetings, or conferences"
            },
            {
                "code": "AG ¶ 34(b)",
                "text": "collecting or storing protected information in any unauthorized location"
            },
            {
                "code": "AG ¶ 34(c)",
                "text": "loading, drafting, editing, modifying, storing, transmitting, or otherwise handling protected information, including images, on any unauthorized equipment or medium"
            },
            {
                "code": "AG ¶ 34(d)",
                "text": "inappropriate efforts to obtain or view protected information outside one's need to know"
            },
            {
                "code": "AG ¶ 34(e)",
                "text": "copying or modifying protected information in an unauthorized manner designed to conceal or remove classification or other document control markings"
            },
            {
                "code": "AG ¶ 34(f)",
                "text": "viewing or downloading information from a secure system when the information is beyond the individual's need-to-know"
            },
            {
                "code": "AG ¶ 34(g)",
                "text": "any failure to comply with rules for the protection of classified or sensitive information"
            },
            {
                "code": "AG ¶ 34(h)",
                "text": "negligence or lax security practices that persist despite counseling by management"
            },
            {
                "code": "AG ¶ 34(i)",
                "text": "failure to comply with rules or regulations that results in damage to the national security, regardless of whether it was deliberate or negligent"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 35(a)",
                "text": "so much time has elapsed since the behavior, or it has happened so infrequently or under such unusual circumstances, that it is unlikely to recur and does not cast doubt on the individual's current reliability, trustworthiness, or good judgment"
            },
            {
                "code": "AG ¶ 35(b)",
                "text": "the individual responded favorably to counseling or remedial security training and now demonstrates a positive attitude toward the discharge of security responsibilities"
            },
            {
                "code": "AG ¶ 35(c)",
                "text": "the security violations were due to improper or inadequate training or unclear instructions"
            },
            {
                "code": "AG ¶ 35(d)",
                "text": "the violation was inadvertent, it was promptly reported, there is no evidence of compromise, and it does not suggest a pattern"
            }
        ]
    },
    "L": {
        "name": "Outside Activities",
        "concern": "Involvement in certain types of outside employment or activities is of security concern if it poses a conflict of interest with an individual's security responsibilities and could create an increased risk of unauthorized disclosure of classified or sensitive information.",
        "disqualifiers": [
            {
                "code": "AG ¶ 37(a)",
                "text": "any employment or service, whether compensated or volunteer, with: (1) the government of a foreign country; (2) any foreign national, organization, or other entity; (3) a representative of any foreign interest; and (4) any foreign, domestic, or international organization or person engaged in analysis, discussion, or publication of material on intelligence, defense, foreign affairs, or protected technology"
            },
            {
                "code": "AG ¶ 37(b)",
                "text": "failure to report or fully disclose an outside activity when this is required"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 38(a)",
                "text": "evaluation of the outside employment or activity by the appropriate security or counterintelligence office indicates that it does not pose a conflict with an individual's security responsibilities or with the national security interests of the United States"
            },
            {
                "code": "AG ¶ 38(b)",
                "text": "the individual terminated the employment or discontinued the activity upon being notified that it was in conflict with his or her security responsibilities"
            }
        ]
    },
    "M": {
        "name": "Use of Information Technology",
        "concern": "Failure to comply with rules, procedures, guidelines, or regulations pertaining to information technology systems may raise security concerns about an individual's reliability and trustworthiness, calling into question the willingness or ability to properly protect sensitive systems, networks, and information.",
        "disqualifiers": [
            {
                "code": "AG ¶ 40(a)",
                "text": "unauthorized entry into any information technology system"
            },
            {
                "code": "AG ¶ 40(b)",
                "text": "unauthorized modification, destruction, or manipulation of, or denial of access to, an information technology system or any data in such a system"
            },
            {
                "code": "AG ¶ 40(c)",
                "text": "use of any information technology system to gain unauthorized access to another system or to a compartmented area within the same system"
            },
            {
                "code": "AG ¶ 40(d)",
                "text": "downloading, storing, or transmitting classified, sensitive, proprietary, or other protected information on or to any unauthorized information technology system"
            },
            {
                "code": "AG ¶ 40(e)",
                "text": "unauthorized use of any information technology system"
            },
            {
                "code": "AG ¶ 40(f)",
                "text": "introduction, removal, or duplication of hardware, firmware, software, or media to or from any information technology system when prohibited by rules, procedures, guidelines, or regulations or when otherwise not authorized"
            },
            {
                "code": "AG ¶ 40(g)",
                "text": "negligence or lax security practices in handling information technology that persists despite counseling by management"
            },
            {
                "code": "AG ¶ 40(h)",
                "text": "any misuse of information technology, whether deliberate or negligent, that results in damage to the national security"
            }
        ],
        "mitigators": [
            {
                "code": "AG ¶ 41(a)",
                "text": "so much time has elapsed since the behavior happened, or it happened under such unusual circumstances, that it is unlikely to recur and does not cast doubt on the individual's reliability, trustworthiness, or good judgment"
            },
            {
                "code": "AG ¶ 41(b)",
                "text": "the misuse was minor and done solely in the interest of organizational efficiency and effectiveness"
            },
            {
                "code": "AG ¶ 41(c)",
                "text": "the conduct was unintentional or inadvertent and was followed by a prompt, good-faith effort to correct the situation and by notification to appropriate personnel"
            },
            {
                "code": "AG ¶ 41(d)",
                "text": "the misuse was due to improper or inadequate training or unclear instructions"
            }
        ]
    }
}


def _index() -> dict[str, dict]:
    """One record per condition code. Duplicate codes are a broken list."""
    out: dict[str, dict] = {}
    for letter, guideline in GUIDELINES.items():
        for kind, key in (("disqualifier", "disqualifiers"), ("mitigator", "mitigators")):
            for item in guideline[key]:
                code = item["code"]
                if code in out:
                    raise RuntimeError(f"duplicate SEAD-4 condition code {code}")
                out[code] = {
                    "code": code,
                    "guideline": letter,
                    "kind": kind,
                    "text": item["text"],
                }
    return out


CONDITIONS: dict[str, dict] = _index()
CODES = frozenset(CONDITIONS)

# 142 is the size of the vendored list (PROJECT_PLAN.md). A shorter or
# longer list means a code was dropped or invented while editing.
if len(CONDITIONS) != 142:
    raise RuntimeError(
        f"SEAD-4 reference must hold 142 conditions, found {len(CONDITIONS)}"
    )
if list(GUIDELINES) != config.GUIDELINE_CODES:
    raise RuntimeError(
        "SEAD-4 guideline letters do not match src.config.GUIDELINE_CODES"
    )

# A condition is a paragraph number plus a subparagraph letter. Bare
# "AG ¶ 19" is the whole paragraph, not a code, and "AG ¶ 190(a)" is not
# a real paragraph. The letter is required so ¶ 2 cannot match inside ¶ 20.
_CITE = re.compile(
    r"(?:AG\s*)?(?:¶|para(?:graph)?\.?)\s*(\d{1,2})\s*\(\s*([a-z])\s*\)",
    re.IGNORECASE,
)


def _as_code(number: str, letter: str) -> str | None:
    code = f"AG ¶ {int(number)}({letter.lower()})"
    if code not in CONDITIONS:
        return None
    return code


def canonical(citation: str) -> str | None:
    """Official code for a citation, or None if it is not in the closed list.

    Accepts the forms the decisions use: "AG ¶ 19(a)", "AG para 19(a)",
    "paragraph 19(a)". The first citation in the string wins. Bare
    "AG ¶ 19", an unknown letter, or an unknown paragraph returns None.
    """
    if not citation:
        return None
    match = _CITE.search(citation)
    if not match:
        return None
    return _as_code(match.group(1), match.group(2))


def condition(citation: str) -> dict | None:
    """The condition record for a citation, or None when the code is not real."""
    code = canonical(citation)
    if code is None:
        return None
    return CONDITIONS[code]


def is_condition(citation: str) -> bool:
    return canonical(citation) is not None


def cited(text: str) -> list[str]:
    """Canonical codes cited in `text`, in order. Invented codes are dropped."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _CITE.finditer(text):
        code = _as_code(match.group(1), match.group(2))
        if code is None or code in seen:
            continue
        seen.add(code)
        found.append(code)
    return found


def conditions_for(letter: str) -> list[dict]:
    """Disqualifiers then mitigators for guideline A-M. Unknown letters are empty."""
    guideline = GUIDELINES.get((letter or "").upper())
    if guideline is None:
        return []
    rows = []
    for key in ("disqualifiers", "mitigators"):
        for item in guideline[key]:
            rows.append(CONDITIONS[item["code"]])
    return rows
