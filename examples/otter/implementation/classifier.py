from typing_extensions import override

from summ.classify.classifier import Classifier, Document

from .classes import MyClasses


class TitleClassifier(Classifier, classes=MyClasses):
    CATEGORY = "JOB_TITLE"
    VARS = {
        "title": "User Interview Title",
        "job_title": "Title Classification",
    }
    EXAMPLES = [
        {
            "title": "Brendan - UiPath - Sales",
            "job_title": MyClasses.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR,
        },
        {
            "title": "Amy - American Express - HR Manager",
            "job_title": MyClasses.JOB_TITLE_MANAGER,
        },
        {
            "title": "Mia - Captial One - Head of IT",
            "job_title": MyClasses.JOB_TITLE_EXECUTIVE,
        },
    ]
    PREFIX = "The user interviewed falls into one of the following categories:"
    SUFFIX = " ".join(
        [
            f"{MyClasses.JOB_TITLE_INDIVIDUAL_CONTRIBUTOR} represents an employee who contributes to a team or organization but do not manage others.",
            f"{MyClasses.JOB_TITLE_MANAGER} represents an employee who takes a leadership role in an organization and manages a team of employees.",
            f"{MyClasses.JOB_TITLE_EXECUTIVE} represents an an employee who has a leadership role in the organization. C-suite members, as well as heads, directors, and VPs of departments count as executives.",
            "Return only a list of industry variables, no explanation. The form of the title is 'Name - Name of Company - Role'. You will have to infer the role from the company name.",
        ]
    )

    @override
    def classify(self, doc: Document) -> dict[str, str]:
        return {"title": doc.metadata["file"]}


class CompanyCategoryClassifier(Classifier, classes=MyClasses):
    CATEGORY = "COMPANY_CATEGORY"
    VARS = {
        "title": "User Interview Title",
        "company_category": "Company Category Classification",
    }
    EXAMPLES = [
        {
            "title": "Brendan - UiPath - Sales",
            "company_category": MyClasses.COMPANY_CATEGORY_RPA,
        },
        {
            "title": "Amy - Accenture - Lead Automation Developer.",
            "company_category": MyClasses.COMPANY_CATEGORY_CONSULTANCY,
        },
        {
            "title": "Mia - Captial One - HR Manager",
            "company_category": MyClasses.COMPANY_CATEGORY_CUSTOMER,
        },
    ]
    PREFIX = "The user interviewed falls into one of the following categories:"
    SUFFIX = " ".join(
        [
            f"{MyClasses.COMPANY_CATEGORY_RPA} represents companies that build RPA software. This includes companies like UiPath and Automation Anywhere.",
            f"{MyClasses.COMPANY_CATEGORY_CUSTOMER} represents a company that would be a buyer of RPA software. They have some course of business that has nothing to do with selling RPA software or services.",
            f"{MyClasses.COMPANY_CATEGORY_CONSULTANCY} represents a company that sells RPA services. There are developers and consultants that help COMPANY_CUSTOMER type companies get RPA up and running.",
            "Return only a list of department variables, no explanation.",
            "The form of the title is 'Name - Name of Company - Role'.",
            "You will have to infer the industry from the company name. Note if a user works at a company has an internal RPA team, they do not neccessarily work at a company that sells RPA software.",
            f"They may work for a {MyClasses.COMPANY_CATEGORY_CUSTOMER} company who has aleady purchased some RPA software in the past.",
        ]
    )

    @override
    def classify(self, doc: Document) -> dict[str, str]:
        return {"title": doc.metadata["file"]}


class DepartmentClassifier(Classifier, classes=MyClasses):
    CATEGORY = "DEPARTMENT"
    VARS = {
        "title": "User Interview Title",
        "department": "Department Classification",
    }
    EXAMPLES = [
        {"title": "Brendan - UiPath - Sales", "department": MyClasses.DEPARTMENT_SALES},
        {
            "title": "Amy - Blue Prism - Lead Automation Developer",
            "department": MyClasses.DEPARTMENT_RPA_DEVELOPMENT,
        },
    ]
    PREFIX = "The user interviewed works in one or more of the following departments:"
    SUFFIX = " ".join(
        [
            "Return only a list of department variables, no explanation.",
            f"For example: '{MyClasses.DEPARTMENT_OPERATIONS}, {MyClasses.DEPARTMENT_SALES}' if the user works in both Operations and Sales.",
            "The form of the title is 'Name - Name of Company - Role'.",
            "Note if a user works at a company that builds RPA software or is an RPA consultancy, they do not work at the RPA Center of Excellence; it means they are an RPA developer to external customers.",
        ]
    )

    @override
    def classify(self, doc: Document) -> dict[str, str]:
        return {"title": doc.metadata["file"]}


class IndustryClassifier(Classifier, classes=MyClasses):
    CATEGORY = "INDUSTRY"
    VARS = {
        "title": "User Interview Title",
        "industry": "Industry Classification",
    }
    EXAMPLES = [
        {
            "title": "Brendan - UiPath - Sales",
            "industry": MyClasses.INDUSTRY_RPA_SOFTWARE,
        },
        {
            "title": "Amy - Accenture - Lead Automation Developer.",
            "industry": MyClasses.INDUSTRY_CONSULTING,
        },
    ]
    PREFIX = "The user interviewed works in one or more of the following industries:"
    SUFFIX = " ".join(
        [
            "Return only a list of department variables, no explanation.",
            f"For example: '{MyClasses.INDUSTRY_SOFTWARE}, {MyClasses.INDUSTRY_HOSPITALITY}' if the user works at a company in the software and hospitality industry.",
            "The form of the title is 'Name - Name of Company - Role'.",
            "You will have to infer the industry from the company name.",
            "Note: if a user works at a company has an internal RPA team, they do not neccessarily work at a company in the RPA Software industry, they may be in the RPA department in a non-RPA company.",
        ]
    )

    @override
    def classify(self, doc: Document) -> dict[str, str]:
        return {"title": doc.metadata["file"]}
