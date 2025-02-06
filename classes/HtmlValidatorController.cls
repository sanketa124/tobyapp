/*
*********************************************************
Apex Class Name    : HtmlValidatorController
Created Date       : Jan 12, 2025
@description       : This class is used for File metadata Upload and Validation Operations.
@author            : Sanket Anand
Modification Log:
Ver   Date         Author                               Modification
1.0   12-01-2025   Sanket Anand                      Initial Version
*********************************************************
*/
public without sharing class HtmlValidatorController {

    /*
    *********************************************************
    @Method Name    : fetchGrammarRules
    @author         : Sanket Anand
    @description    : Method to fetch all the DFA rules to match file against
    @param          : void
    @return         : List of Rules in JSON Serialized form
    ********************************************************
    */
    @AuraEnabled(cacheable=true)
    public static String fetchGrammarRules() {
        List<DFA_Rule__c> rules = [
            SELECT Id, Regex_Expression__c, Description__c, Name 
            FROM DFA_Rule__c
        ];
        return JSON.serialize(rules);
    }

    /*
    *********************************************************
    @Method Name    : getTextFileContent
    @author         : Sanket Anand
    @description    : Method to get contents of the file uploaded
    @param          : contentVersionId: Id of file
    @return         : Binary content of file
    ********************************************************
    */
    @AuraEnabled(cacheable=true)
    public static String getTextFileContent(String contentVersionId) {

        //Check if file ID is not blank
        if (String.isEmpty(contentVersionId)) {
            throw new IllegalArgumentException('Content Version ID cannot be null or empty.');
        }

        // Query the Content Version associated with the Content Document ID
        ContentVersion contentVersion = [
            SELECT Id, Title, FileType, VersionData
            FROM ContentVersion
            WHERE Id =: contentVersionId
        ];

        // Retrieve and return the file content as a String
        Blob fileBlob = contentVersion.VersionData;
        return fileBlob.toString();
    }

    /*
    *********************************************************
    @Method Name    : createFileLink
    @author         : Sanket Anand
    @description    : Method to create user
    @param          : fileId: ID of file, rulesMatched: IDs of rules matched, finalResult: JSON with matched rules details
    @return         : void
    ********************************************************
    */
    @AuraEnabled(cacheable=false)
    public static void createFileLink(String fileId, String[] rulesMatched, String finalResult) {

        //Check if File ID is present
        if (String.isEmpty(fileId)) {
            throw new IllegalArgumentException('File ID cannot be null or empty.');
        }
        
        //Create Parent File Tag for matched rules
        List<HtmlTagRecord__c> htcs = new List<HtmlTagRecord__c>();
        HtmlTagRecord__c ht = new HtmlTagRecord__c();
        ht.Tags__c = finalResult;
        insert ht;

        //Link file Tags with file.
        List<ContentDocumentLink> cdl = new List<ContentDocumentLink>();
        ContentDocumentLink cd0 = new ContentDocumentLink();
        cd0.LinkedEntityId = ht.Id;
        cd0.ContentDocumentId = fileId;
        cdl.add(cd0);

        //Create Child File Tags for matched rules
        for(String s: rulesMatched){
            HtmlTagRecord__c htc = new HtmlTagRecord__c();
            htc.Tags__c = finalResult;
            htc.DFA_Rule__c = s;
            htc.Parent_Record__c = ht.Id;
            htcs.add(htc);
            ContentDocumentLink cd = new ContentDocumentLink();
            cd.LinkedEntityId = s;
            cd.ContentDocumentId = fileId;
            cdl.add(cd);
        }

        //Insert child file tags
        if(htcs.size() > 0){
            insert htcs;
        }

        //Insert child file tag links to the file
        if(cdl.size() > 0){
            insert cdl;
        }
    }
}