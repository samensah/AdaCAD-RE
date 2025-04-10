import torch

def refind_relation_descriptions(subject, object):
    """Define relation descriptions using actual subject and object entity text"""
    return {
        "no_relation": f"There is no specific relationship between {subject} and {object} or the relation is not covered by other categories",
        "pers:title:title": f"{subject} holds or has held {object} title (job position or role)",
        "org:gpe:operations_in": f"{subject} operates or has operated in {object}",
        "pers:org:employee_of": f"{subject} is or was an employee of {object}",
        "org:org:agreement_with": f"{subject} has or had an agreement, partnership, or contract with {object}",
        "org:date:formed_on": f"{subject} was formed or founded on {object}",
        "pers:org:member_of": f"{subject} is or was a member of {object}",
        "org:org:subsidiary_of": f"{subject} is or was a subsidiary of {object}",
        "org:org:shares_of": f"{subject} owns or owned shares of {object}",
        "org:money:revenue_of": f"{object} is or was the revenue of {subject}",
        "org:money:loss_of": f"{object} is or was the loss reported by {subject}",
        "org:gpe:headquartered_in": f"{subject} is or was headquartered in {object}",
        "org:date:acquired_on": f"{subject} was acquired on {object}",
        "pers:org:founder_of": f"{subject} is or was the founder of {object}",
        "org:gpe:formed_in": f"{subject} was formed or founded in {object}",
        "org:org:acquired_by": f"{subject} was acquired by {object}",
        "pers:univ:employee_of": f"{subject} is or was an employee of {object}",
        "pers:gov_agy:member_of": f"{subject} is or was a member of {object}",
        "pers:univ:attended": f"{subject} attended or has attended {object}",
        "pers:univ:member_of": f"{subject} is or was a member of {object}",
        "org:money:profit_of": f"{object} is or was the profit of {subject}",
        "org:money:cost_of": f"{object} is or was the cost incurred by {subject}"
    }

refind_relation_regex = r"(no_relation|pers:title:title|org:gpe:operations_in|pers:org:employee_of|org:org:agreement_with|org:date:formed_on|pers:org:member_of|org:org:subsidiary_of|org:org:shares_of|org:money:revenue_of|org:money:loss_of|org:gpe:headquartered_in|org:date:acquired_on|pers:org:founder_of|org:gpe:formed_in|org:org:acquired_by|pers:univ:employee_of|pers:gov_agy:member_of|pers:univ:attended|pers:univ:member_of|org:money:profit_of|org:money:cost_of)"
    

def biored_relation_descriptions(subject, object):
    """Define relation descriptions using actual subject and object entity text"""
    return {
        "Association": f"Relation where {subject} is associated with {object}",
        "Bind": f"Relation where {subject} binds to {object}",
        "Comparison": f"Relation where {subject} is compared with {object}",
        "Conversion": f"Relation where {subject} is converted to {object}",
        "Cotreatment": f"Relation where {subject} is used in treatment along with {object}",
        "Drug_Interaction": f"Relation where {subject} interacts with {object}",
        "Negative_Correlation": f"Relation where {subject} is negatively correlated with {object}",
        "Positive_Correlation": f"Relation where {subject} is positively correlated with {object}"
    }
biored_relation_regex = r"(Association|Bind|Comparison|Conversion|Cotreatment|Drug_Interaction|Negative_Correlation|Positive_Correlation)"

def tacred_relation_descriptions(subject, object):
    return  {
        "no_relation": f"There is no specific relation between {subject} and {object}",
        "org:alternate_names": f"{object} is an alternate name for {subject}",
        "org:city_of_headquarters": f"{subject} has its headquarters in the city of {object}",
        "org:country_of_headquarters": f"{subject} has its headquarters in the country of {object}",
        "org:dissolved": f"{subject} was dissolved on {object}",
        "org:founded": f"{subject} was founded on {object}",
        "org:founded_by": f"{subject} was founded by {object}",
        "org:member_of": f"{subject} is a member of {object}",
        "org:members": f"{object} is a member of {subject}",
        "org:number_of_employees/members": f"{subject} has {object} employees or members",
        "org:parents": f"{object} is a parent organization of {subject}",
        "org:political/religious_affiliation": f"{subject} has a political or religious affiliation with {object}",
        "org:shareholders": f"{object} is a shareholder of {subject}",
        "org:stateorprovince_of_headquarters": f"{subject} has its headquarters in the state or province of {object}",
        "org:subsidiaries": f"{object} is a subsidiary of {subject}",
        "org:top_members/employees": f"{object} is a top member or employee of {subject}",
        "org:website": f"{object} is the website of {subject}",
        "per:age": f"{subject} is {object} years old",
        "per:alternate_names": f"{object} is an alternate name for {subject}",
        "per:cause_of_death": f"{subject} died because of {object}",
        "per:charges": f"{subject} has been charged with {object}",
        "per:children": f"{object} is a child of {subject}",
        "per:cities_of_residence": f"{subject} lives or has lived in the city of {object}",
        "per:city_of_birth": f"{subject} was born in the city of {object}",
        "per:city_of_death": f"{subject} died in the city of {object}",
        "per:countries_of_residence": f"{subject} lives or has lived in the country of {object}",
        "per:country_of_birth": f"{subject} was born in the country of {object}",
        "per:country_of_death": f"{subject} died in the country of {object}",
        "per:date_of_birth": f"{subject} was born on {object}",
        "per:date_of_death": f"{subject} died on {object}",
        "per:employee_of": f"{subject} is or was an employee of {object}",
        "per:origin": f"{subject} has an ethnic or national origin of {object}",
        "per:other_family": f"{subject} has other family relationship with {object}",
        "per:parents": f"{object} is a parent of {subject}",
        "per:religion": f"{subject} follows the religion of {object}",
        "per:schools_attended": f"{subject} attended {object}",
        "per:siblings": f"{object} is a sibling of {subject}",
        "per:spouse": f"{object} is a spouse of {subject}",
        "per:stateorprovince_of_birth": f"{subject} was born in the state or province of {object}",
        "per:stateorprovince_of_death": f"{subject} died in the state or province of {object}",
        "per:stateorprovinces_of_residence": f"{subject} lives or has lived in the state or province of {object}",
        "per:title": f"{subject} has the title of {object}"
    }

tacred_relation_regex = r'(no_relation|org:(alternate_names|city_of_headquarters|country_of_headquarters|dissolved|founded|founded_by|member_of|members|number_of_employees/members|parents|political/religious_affiliation|shareholders|stateorprovince_of_headquarters|subsidiaries|top_members/employees|website)|per:(age|alternate_names|cause_of_death|charges|children|cities_of_residence|city_of_birth|city_of_death|countries_of_residence|country_of_birth|country_of_death|date_of_birth|date_of_death|employee_of|origin|other_family|parents|religion|schools_attended|siblings|spouse|stateorprovince_of_birth|stateorprovince_of_death|stateorprovinces_of_residence|title))'
