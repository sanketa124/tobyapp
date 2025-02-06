/*
*********************************************************
File Name    : TobyFileUploadAndValidator.js
Created Date       : Jan 12, 2025
@description       : This JS class is used for File metadata Upload, Validation and Replace Operations.
@author            : Sanket Anand
Modification Log:
Ver   Date         Author                               Modification
1.0   12-01-2025   Sanket Anand                      Initial Version
*********************************************************
*/
import { LightningElement, track } from 'lwc';
import fetchGrammarRules from '@salesforce/apex/HtmlValidatorController.fetchGrammarRules';
import getTextFileContent from '@salesforce/apex/HtmlValidatorController.getTextFileContent';
import createFileLink from '@salesforce/apex/HtmlValidatorController.createFileLink';

export default class TobyFileUploadAndValidator extends LightningElement {
    @track validationResult = ''; // Validation rule triggered (If any)
    @track compliantRules = []; //Set of rules matched in the file
    @track fileContent = ''; // Binary content of the file uploaded
    @track modifiedContent = ""; // Stores the modified file content
    replacementText = ""; // User-provided replacement text
    rules = []; // Set of all rules to match with
    value = ''; // Replacement text provided by the user
    contentDocumentID; //ID of File uploaded
    options = [{ label: 'Choose One...', value: '' }]; //Values for the Dropdown for rule to replace

    /*
    *********************************************************
    @Method Name    : compliantRulesExists
    @author         : Sanket Anand
    @description    : Getter function to check if compliant rules exist or not.
    ********************************************************
    */
    get compliantRulesExists(){
        return ((this.compliantRules.length > 0) || (this.fileContent.length > 0));
    }

    /*
    *********************************************************
    @Method Name    : handleChange
    @author         : Sanket Anand
    @description    : Method to select rule to replace in text 
    ********************************************************
    */
    handleChange(event) {
        this.value = event.detail.value;
    }    

    /*
    *********************************************************
    @Method Name    : connectedCallback
    @author         : Sanket Anand
    @description    : Method to fetch all the DFA rules to match file against
    ********************************************************
    */
    connectedCallback() {
        // Fetch grammar rules on component load
        fetchGrammarRules()
            .then((result) => {
                this.rules = JSON.parse(result);
            })
            .catch((error) => {
                console.error('Error fetching grammar rules:', error);
            });
    }

    /*
    *********************************************************
    @Method Name    : handleClick
    @author         : Sanket Anand
    @description    : Method to validated uploaded file against the rules
    ********************************************************
    */
    handleClick(){
        const regex = new RegExp(this.value, "g"); // Create regex with global flag
        this.fileContent = this.fileContent.replace(regex, this.replacementText); // Replace text
        this.validateHtml(this.fileContent);
    }

    /*
    *********************************************************
    @Method Name    : handleFileUpload
    @author         : Sanket Anand
    @description    : Method to fetch and validate binary content of the file content
    ********************************************************
    */
    handleFileUpload(event) {
        this.options = [{ label: 'Choose One...', value: '' }];
        const uploadedFiles = event.detail.files;
        if (uploadedFiles && uploadedFiles.length > 0) {
            const file = uploadedFiles[0];
            this.contentDocumentID = file.documentId;
            getTextFileContent({contentVersionId: file.contentVersionId}).then((resultApex) => {
                this.fileContent = resultApex; // Store the file content for preview
                this.validateHtml(resultApex);
            });
        } else {
            this.validationResult = 'No file was uploaded.';
        }
    }

    /*
    *********************************************************
    @Method Name    : handleReplacementChange
    @author         : Sanket Anand
    @description    : Method to update text with the replacement text entered
    ********************************************************
    */
    handleReplacementChange(event) {
        this.replacementText = event.target.value;
    }

    /*
    *********************************************************
    @Method Name    : validateHtml
    @author         : Sanket Anand
    @description    : Method to validate HTML with standard DOMParser API
    ********************************************************
    */
    validateHtml(content) {
        try {
            const parser = new DOMParser();
            const doc = parser.parseFromString(content, 'text/html');
            const errors = doc.querySelectorAll('parsererror');

            if (errors.length > 0) {
                this.validationResult = `Invalid HTML. Found ${errors.length} error(s).`;
            } else {
                this.validationResult = 'The HTML is valid.';
                this.checkGrammarCompliance(content);
            }
        } catch (error) {
            this.validationResult = `Error parsing HTML: ${error.message}`;
        }
    }

    /*
    *********************************************************
    @Method Name    : checkGrammarCompliance
    @author         : Sanket Anand
    @description    : Method to check for compliant grammar rules and call backend function for creating linkeages 
    ********************************************************
    */
    checkGrammarCompliance(htmlContent) {
        const compliantRules = this.rules.map((rule) => {
            const regex = new RegExp(rule.Regex_Expression__c, 'g');
            const matches = htmlContent.match(regex) || [];
            return {
                ...rule,
                matchCount: matches.length, // Count the matches
            };
        }).filter((rule) => rule.matchCount > 0); // Only include rules with matches
        const applicableRule = compliantRules.map((rule) => {
            this.options.push({label: rule.Name, value: rule.Regex_Expression__c});
            return rule.Id;
        });
        this.compliantRules = compliantRules;
        if (compliantRules.length === 0) {
            this.validationResult += ' No grammar rules matched.';
        } else {
            this.validationResult += ` Complies with ${compliantRules.length} grammar rule(s).`;
            createFileLink({fileId: this.contentDocumentID, rulesMatched: applicableRule, finalResult: JSON.stringify(this.compliantRules)}).then(() => {
                console.log('records saved.')
            });
        }
    }
}