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